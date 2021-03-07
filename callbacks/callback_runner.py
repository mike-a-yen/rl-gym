class CallbackRunner:
    def __init__(self, *callbacks, **kwargs) -> None:
        for name, val in kwargs.items():
            setattr(self, name, val)

        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.bind(self)

    def __call__(self, event_name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            callback(event_name)(*args, **kwargs)
