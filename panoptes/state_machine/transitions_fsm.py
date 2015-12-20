import os
import sys
import yaml
import transitions
import asyncio
import signal
from functools import partial

from ..utils.logger import has_logger
from ..utils.database import PanMongo
from ..utils.modules import load_module
from ..utils import error, listify


@has_logger
class PanStateMachine(transitions.Machine):

    """ A finite state machine for PANOPTES.

    The state machine guides the overall action of the unit. The state machine works in the following
    way with PANOPTES::

            * The machine consists of `states` and `transitions`.
    """

    def __init__(self, *args, **kwargs):
        assert 'states' in kwargs, self.logger.warning('states keyword required.')
        assert 'transitions' in kwargs, self.logger.warning('transitions keyword required.')

        # Set up connection to database
        if not hasattr(self, 'db'):
            self.db = PanMongo()

        try:
            self.state_information = self.db.state_information
        except AttributeError as err:
            raise error.MongoCollectionNotFound(
                msg="Can't connect to mongo instance for states information table. {}".format(err))

        # For tracking the state information
        self._state_stats = dict()

        self._initial = kwargs.get('initial', 'sleeping')

        self._transitions = kwargs['transitions']
        self._states = kwargs['states']

        self.transitions = [self._load_transition(transition) for transition in self._transitions]
        self.states = [self._load_state(state) for state in self._states]

        # Get the asyncio loop
        self.logger.debug("Getting event loop for state machine")
        self._loop = asyncio.get_event_loop()
        self._loop_delay = kwargs.get('loop_delay', 5)  # Default delay

        # Setup utils for graceful shutdown
        self.logger.info("Setting up interrupt handlers for state machine")
        for sig in ('SIGINT', 'SIGTERM'):
            self._loop.add_signal_handler(getattr(signal, sig), partial(self._sigint_handler))

        super().__init__(
            states=self.states,
            transitions=self.transitions,
            initial=self._initial,
            send_event=True,
            before_state_change='before_state',
            after_state_change='after_state'
        )

        self.logger.info("State machine created")

##################################################################################################
# Properties
##################################################################################################

    @property
    def next_state(self):
        """ str: name of next state in state machine """
        return self._next_state

    @next_state.setter
    def next_state(self, next_state):
        self._next_state = next_state

    @property
    def prev_state(self):
        """ str: name of prev state in state machine """
        return self._prev_state

    @prev_state.setter
    def prev_state(self, prev_state):
        self._prev_state = prev_state


##################################################################################################
# Methods
##################################################################################################
    def run(self):
        """ Runs the event loop

        This method starts the main asyncio event loop and stays in the loop until a SIGINT or
        SIGTERM is received (see `_sigint_handler`)
        """
        try:
            self.logger.debug("Starting event loop")
            self._loop.run_forever()
            self.logger.debug("Event loop stopped")
        finally:
            self.logger.debug("Closing event loop")
            self._loop.close()

##################################################################################################
# Callback Methods
##################################################################################################

    def before_state(self, event_data):
        """ Called before each state.

        Starts collecting stats on this particular state, which are saved during
        the call to `after_state`.

        Args:
            event_data(transitions.EventData):  Contains informaton about the event
         """
        self.logger.debug("Before going {} from {}".format(event_data.state.name, event_data.event.name))

        # self._state_stats = dict()
        # self._state_stats['state'] = event_data.state.name
        # self._state_stats['from'] = event_data.event.name.replace('to_', '')
        # self._state_stats['start_time'] = datetime.datetime.utcnow()

    def after_state(self, event_data):
        """ Called after each state.

        Updates the mongodb collection for state stats.

        Args:
            event_data(transitions.EventData):  Contains informaton about the event
        """
        self.logger.debug("After going {} from {}".format(event_data.event.name, event_data.state.name))

        # self._state_stats['stop_time'] = datetime.datetime.utcnow()
        # self.state_information.insert(self._state_stats)

    def execute(self, event_data):
        """ Executes the main data for the state.

        After executing main function, check return state for validitiy. If 'exit'
        state is received for `next_state`, begin to exit system.

        Args:
            event_data(transitions.EventData):  Contains informaton about the event.

        Note:
            This method doesn't return anything but does set the `next_state` and `prev_state` properties.
        """
        self.logger.debug("Inside {} state".format(event_data.state.name))

        # Default next state
        next_state_name = 'parking'

        # Run the `main` method for the state. Every state is required to implement this method.
        try:
            next_state_name = event_data.state.main(event_data)
        except AssertionError as err:
            self.logger.warning("Make sure the mount is initialized: {}".format(err))
        except Exception as e:
            self.logger.warning(
                "Problem calling `main` for state {}: {}".format(event_data.state.name, e))

        if next_state_name in self._states:
            self.logger.debug("{} returned {}".format(event_data.state.name, next_state_name))
            self.next_state = next_state_name
            self.prev_state = event_data.state.name

        if next_state_name == 'exit':
            self.logger.warning("Received exit signal")
            self.next_state = next_state_name
            self.prev_state = event_data.state.name

        self.logger.debug("Next state is: {}".format(self.next_state))


##################################################################################################
# Class Methods
##################################################################################################

    @classmethod
    def load_state_table(cls, state_table_name='simple_state_table'):
        """ Loads the state table

        Args:
            state_table_name(str):  Name of state table. Corresponds to file name in
                `$POCS/resources/state_table/` directory. Default 'simple_state_table'.

        Returns:
            dict:                   Dictonary with `states` and `transitions` keys.
        """

        state_table_file = "{}/resources/state_table/{}.yaml".format(
            os.getenv('POCS', default='/var/panoptes/POCS'), state_table_name)

        state_table = {'states': [], 'transitions': []}

        try:
            with open(state_table_file, 'r') as f:
                state_table = yaml.load(f.read())
        except OSError as err:
            raise error.InvalidConfig(
                'Problem loading state table yaml file: {} {}'.format(err, state_table_file))
        except:
            raise error.InvalidConfig(
                'Problem loading state table yaml file: {}'.format(state_table_file))

        return state_table

##################################################################################################
# Private Methods
##################################################################################################

    def _sigint_handler(self, signum, frame):
        """
        Interrupt signal handler. Designed to intercept a Ctrl-C from
        the user and properly shut down the system.
        """
        self.logger.error("Signal handler called with signal {}".format(signum))
        try:
            self._loop.stop()
            self.power_down()
        except Exception as e:
            self.logger.error("Problem powering down. PLEASE MANUALLY INSPECT THE MOUNT.")
            self.logger.error("Error: {}".format(e))
        finally:
            sys.exit(0)

    def _load_state(self, state):
        self.logger.debug("Loading {} state".format(state))
        state_module = load_module('panoptes.state_machine.states.{}'.format(state))

        return state_module.State(name=state, panoptes=self)

    def _load_transition(self, transition):
        self.logger.debug("Loading transition: {}".format(transition))

        # # Make sure the transition has the weather_is_safe condition on it
        # conditions = listify(transition.get('conditions', []))

        # conditions.append('is_safe')
        # transition['conditions'] = conditions

        self.logger.debug("Returning transition: {}".format(transition))
        return transition
