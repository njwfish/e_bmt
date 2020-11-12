def create_register_fn(registry):
    def named_register(name):
        def register(fn):
            registry[name] = fn
            fn.name = name
            return fn

        return register

    return named_register
