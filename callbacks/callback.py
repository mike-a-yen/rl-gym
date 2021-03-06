import logging
import os

log = logging.getLogger(__file__)


class Callback:
    def bind(self, runner) -> None:
        self.runner = runner
        setattr(self.runner, self.name, self)

    def __call__(self, event_name: str):
        if not self.is_bound:
            log.warn(f'Callback {self.name} is not bound to a CallbackRunner.')
        callback_fn = getattr(self, event_name)
        return callback_fn

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_episode_begin(self):
        pass

    def on_episode_end(self) -> None:
        pass

    def on_action_begin(self, state, Q):
        pass
    
    def on_action_end(self, action):
        pass

    def on_step_begin(self, action, was_random):
        pass

    def on_step_end(self, state, action, reward, next_state, done, was_random) -> None:
        pass

    def on_eval_begin(self):
        pass

    def on_eval_end(self):
        pass

    @property
    def is_bound(self) -> bool:
        return getattr(self, 'runner', None) is not None

    @property
    def name(self):
        return self.__class__.__name__
