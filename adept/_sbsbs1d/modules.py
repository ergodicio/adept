import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from equinox import tree_at
import sys
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
        
class CNN(eqx.Module):
    layers: list
    final_size: int
    def __init__(self,in_channels,in_size,out_features):
        seed = np.random.randint(0,int(1e4))
        key = jax.random.PRNGKey(seed)
        channel_progression = [in_channels, in_channels, in_channels, in_channels]
        self.final_size = in_size*in_channels//8
        self.layers = [
            eqx.nn.Conv1d(in_channels=channel_progression[0],out_channels=channel_progression[1],kernel_size=4,stride=2,padding=1,key=key),
            jax.nn.relu,
            eqx.nn.Conv1d(in_channels=channel_progression[1],out_channels=channel_progression[2],kernel_size=4,stride=2,padding=1,key=key),
            jax.nn.relu,
            eqx.nn.Conv1d(in_channels=channel_progression[2],out_channels=channel_progression[3],kernel_size=4,stride=2,padding=1,key=key),
            jax.nn.relu,
            eqx.nn.Linear(in_features=self.final_size,out_features=out_features,key=key)
        ]
        
    def __call__(self,x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x.reshape(self.final_size))
        return x
        


class HohlNet(eqx.Module):
    nn_input_laser_leh: CNN
    nn_mlp_leh: eqx.Module
    nn_input_laser_hohl: CNN
    nn_mlp_hohl: eqx.Module
    nn_input_laser_sbs_source: CNN
    nn_mlp_sbs_source: eqx.Module
    leh_outputs: list
    hohl_outputs: list
    sbs_source_outputs: list
    
    def __init__(self,beams,t_pts,n_design_params):
        seed = np.random.randint(0,int(1e4))
        key = jax.random.PRNGKey(seed)
        cnn_out_features = 64
        depth = 3
        width = 16
        
        self.nn_input_laser_leh = CNN(beams,t_pts,cnn_out_features)
        self.nn_mlp_leh = eqx.nn.MLP(cnn_out_features+n_design_params+1,9,width_size=width,depth=depth,key=key)
        self.leh_outputs = ['n_over_n0','Te_over_T0','Ti_over_T0','flow_magnitude','flow_theta','flow_phi','Zeff','A_ion','L_p']
        #inputs = embedded input pulse, design inputs, t
        #outputs = n, Te, Ti, flow_magnitude, flow_theta, flow_phi, Zeff, A_ion, Lp
        
        self.nn_input_laser_hohl = CNN(beams,t_pts,cnn_out_features)
        self.nn_mlp_hohl = eqx.nn.MLP(cnn_out_features+n_design_params+3,7,width_size=width,depth=depth,key=key)
        self.hohl_outputs = ['n_over_n0','Te_over_T0','Ti_over_T0','flow_over_flow0','Zeff','A_ion','omegabeat']
        #inputs = embedded input pulse, design inputs, subcone index, z, t
        #outputs = n, Te, Ti, flow (along beam), Zeff, A_ion, omegabeat
        
        self.nn_input_laser_sbs_source = CNN(beams,t_pts,cnn_out_features)
        self.nn_mlp_sbs_source = eqx.nn.MLP(cnn_out_features+n_design_params+2,1,width_size=width,depth=depth,key=key)
        self.sbs_source_outputs = ['thermal_noise']
        #inputs = embedded input pulse, design inputs, subcone index, t
        #outputs = thermal_noise

    def leh_eval(self,input_powers,design_inputs,time):
        embedded_input_laser = self.nn_input_laser_leh(input_powers)
        x_in = jnp.concatenate([embedded_input_laser,design_inputs,time],axis=0)
        leh_plasma = self.nn_mlp_leh(x_in)
        leh_plasma_dict = {self.leh_outputs[i]:leh_plasma[i] for i in range(len(self.leh_outputs))}
        return leh_plasma_dict
    
    def hohl_eval(self,input_powers,design_inputs,subcone_index,z,t):
        embedded_input_laser = self.nn_input_laser_hohl(input_powers)
        x_in = jnp.concatenate([embedded_input_laser,design_inputs,subcone_index,z,t],axis=0)
        hohl_plasma = self.nn_mlp_hohl(x_in)
        hohl_plasma_dict = {self.hohl_outputs[i]:hohl_plasma[i] for i in range(len(self.hohl_outputs))}
        return hohl_plasma_dict
    
    def sbs_source_eval(self,input_powers,design_inputs,subcone_index,t):
        embedded_input_laser = self.nn_input_laser_sbs_source(input_powers)
        x_in = jnp.concatenate([embedded_input_laser,design_inputs,subcone_index,t],axis=0)
        sbs_source = self.nn_mlp_sbs_source(x_in)
        sbs_source_dict = {self.sbs_source_outputs[i]:sbs_source[i] for i in range(len(self.sbs_source_outputs))}
        return sbs_source_dict
    
    def get_partition_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)
        nn_input_laser_leh_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_leh)
        for i,layer in enumerate(self.nn_input_laser_leh.layers):
            if hasattr(layer,"weight"):
                nn_input_laser_leh_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_leh_filter_spec,
                    replace=(True, True)
                )
            
        nn_mlp_leh_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_leh)
        for i,layer in enumerate(self.nn_mlp_leh.layers):
            if hasattr(layer,"weight"):
                nn_mlp_leh_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_leh_filter_spec,
                    replace=(True, True)
                )  
                  
        nn_input_laser_hohl_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_hohl)
        for i,layer in enumerate(self.nn_input_laser_hohl.layers):
            if hasattr(layer,"weight"):
                nn_input_laser_hohl_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_hohl_filter_spec,
                    replace=(True, True)
                )  
                  
        nn_mlp_hohl_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_hohl)
        for i,layer in enumerate(self.nn_mlp_hohl.layers):
            if hasattr(layer,"weight"):
                nn_mlp_hohl_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_hohl_filter_spec,
                    replace=(True, True)
                )    
                
        nn_input_laser_sbs_source_filter_spec = jtu.tree_map(lambda _: False, self.nn_input_laser_sbs_source)
        for i,layer in enumerate(self.nn_input_laser_sbs_source.layers):
            if hasattr(layer,"weight"):
                nn_input_laser_sbs_source_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_input_laser_sbs_source_filter_spec,
                    replace=(True, True)
                )  
                  
        nn_mlp_sbs_source_filter_spec = jtu.tree_map(lambda _: False, self.nn_mlp_sbs_source)
        for i,layer in enumerate(self.nn_mlp_sbs_source.layers):
            if hasattr(layer,"weight"):
                nn_mlp_sbs_source_filter_spec = tree_at(
                    lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
                    nn_mlp_sbs_source_filter_spec,
                    replace=(True, True)
                )        
        
        filter_spec = tree_at(lambda tree: tree.nn_input_laser_leh, filter_spec, replace=nn_input_laser_leh_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_mlp_leh, filter_spec, replace=nn_mlp_leh_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_input_laser_hohl, filter_spec, replace=nn_input_laser_hohl_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_mlp_hohl, filter_spec, replace=nn_mlp_hohl_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_input_laser_sbs_source, filter_spec, replace=nn_input_laser_sbs_source_filter_spec)
        filter_spec = tree_at(lambda tree: tree.nn_mlp_sbs_source, filter_spec, replace=nn_mlp_sbs_source_filter_spec)
        
        # amp_model_filter_spec = jtu.tree_map(lambda _: False, self.amp_model)
        # for i in range(len(self.amp_model.layers)):
        #     amp_model_filter_spec = tree_at(
        #         lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
        #         amp_model_filter_spec,
        #         replace=(True, True),
        #     )

        # phase_model_filter_spec = jtu.tree_map(lambda _: False, self.phase_model)
        # for i in range(len(self.phase_model.layers)):
        #     phase_model_filter_spec = tree_at(
        #         lambda tree: (tree.layers[i].weight, tree.layers[i].bias),
        #         phase_model_filter_spec,
        #         replace=(True, True),
        #     )

        # filter_spec = tree_at(lambda tree: tree.phase_model, filter_spec, replace=phase_model_filter_spec)
        # filter_spec = tree_at(lambda tree: tree.amp_model, filter_spec, replace=amp_model_filter_spec)

        return filter_spec