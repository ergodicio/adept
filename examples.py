# Class-based
class Lagrangian1dModule():
    ...
    def log_base_lrh1d_stuff():
        pass


class BaseLagrangian1DModule(Lagrangian1DModule):
    ...


class SodShocktubeModule(Lagrangian1DModule):
    def call(self, run_id):
        result = super().call()
        my_metric = ...
        return {"result": result, "my_metric": my_metric}

    def postprocess(self, result, run_id):
        #super().postprocess()
        log_base_lrh1d_stuff()
        log_my_metric(run_id, result["my_metric"])







# function-based

# Basic call
@jax.jit
run_id = create_run_id()
prelogging(run_id)
setup_result = general_lagrangian1d_setup(cfg)
diffrax_result = call_lagrangian_1d(setup_result)
postprocess(diffrax_result, setup_result, run_id)



# Sod shocktube call
@jax.jit
run_id = create_run_id()
prelogging(run_id)

