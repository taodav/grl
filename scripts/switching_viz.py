import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from grl.utils.file_system import load_info


class MemDataCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, name: str, color: str, data: np.ndarray,
                 pt_size: int = 5,
                 always_visible: bool = False):
        self.name = name
        self.color = color

        self.data = data
        self.visibility = True
        self.always_visible = always_visible
        self.actor = None
        self.curr_idx = 0
        self.pt_size = pt_size

    def __call__(self, state: bool):
        # if self.actor is None:
        #     pc = pv.PolyData(self.data)
        #     self.actor = plotter.add_mesh(pc, color=self.color, point_size=5, name=self.name)
        self.actor.SetVisibility(state)
        self.visibility = state


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='Path to the .npy file that is outputted by value_and_mem_polytope.py.')
    args = parser.parse_args()

    res = load_info(args.path)
    objective = res['objective']
    vals_term_removed = res['values_terminal_removed']
    mem_vals_term_removed = res['memory_values_terminal_removed']
    all_mem_params = res['all_mem_params']

    # Now we plot our value functions
    plotter = pv.Plotter()

    plotter.add_title(objective)

    state_val_cloud = pv.PolyData(vals_term_removed['state_v'])
    plotter.add_mesh(state_val_cloud, color='gray', point_size=3, opacity=0.01,
                     label='pi(s)')

    pi_obs_state_val_cloud = pv.PolyData(vals_term_removed['pi_obs_state_v'])
    plotter.add_mesh(pi_obs_state_val_cloud, point_size=5, color='yellow',
                     label='pi(o)')


    if objective in ['ld', 'mem_state_discrep']:
        all_mem_vals_info = [
            MemDataCallback(name='mem_0_mc', color='blue', data=mem_vals_term_removed['mem_0_state_mc_v']),
            MemDataCallback(name='mem_1_mc', color='cyan', data=mem_vals_term_removed['mem_1_state_mc_v']),
            MemDataCallback(name='mem_0_td', color='red', data=mem_vals_term_removed['mem_0_state_td_v']),
            MemDataCallback(name='mem_1_td', color='orange', data=mem_vals_term_removed['mem_1_state_td_v']),
            MemDataCallback(name='mem_0_mc_init_pi', color='blue', pt_size=10,
                            data=mem_vals_term_removed['mem_0_state_mc_v_fixed_pi'], always_visible=True),
            MemDataCallback(name='mem_1_mc_init_pi', color='cyan', pt_size=10,
                            data=mem_vals_term_removed['mem_1_state_mc_v_fixed_pi'], always_visible=True),
            MemDataCallback(name='mem_0_td_init_pi', color='red', pt_size=10,
                            data=mem_vals_term_removed['mem_0_state_td_v_fixed_pi'], always_visible=True),
            MemDataCallback(name='mem_1_td_init_pi', color='orange', pt_size=10,
                            data=mem_vals_term_removed['mem_1_state_td_v_fixed_pi'], always_visible=True),
        ]
    elif objective == 'tde':
        all_mem_vals_info = [
            MemDataCallback(name='one_step_mem_0_td', color='blue', data=mem_vals_term_removed['one_step_mem_0_state_td_v']),
            MemDataCallback(name='one_step_mem_1_td', color='cyan', data=mem_vals_term_removed['one_step_mem_1_state_td_v']),
            MemDataCallback(name='mem_0_td', color='red', data=mem_vals_term_removed['mem_0_state_td_v']),
            MemDataCallback(name='mem_1_td', color='orange', data=mem_vals_term_removed['mem_1_state_td_v']),
        ]

    # TODO: What we can do is add_mesh for all time, and just call SetVisibility to False for everything except for current slice.

    def create_mesh(value, widget):
        size = 50
        startpos = 12

        # use 'value' to adjust the number of points plotted
        idx = int(value)
        for d in all_mem_vals_info:
            pc = pv.PolyData(d.data[idx])
            d.idx = idx
            if d.visibility or d.always_visible:
                actor = plotter.add_mesh(pc, color=d.color, point_size=d.pt_size, name=d.name)
                d.actor = actor
            else:
                d.actor = None

            if not d.always_visible:
                plotter.add_checkbox_button_widget(
                    d,
                    value=d.visibility,
                    position=(5.0, startpos),
                    size=size,
                    border_size=1,
                    color_on=d.color,
                    color_off='grey',
                    background_color='grey',
                )
                plotter.add_text(
                    d.name,
                    position=(5 + 60, startpos)
                )
            startpos = startpos + size + (size // 10)
        # plotter.render()

    slider_widget = plotter.add_slider_widget(create_mesh, [0, all_mem_params.shape[0] - 1], value=0, title='Update Number', pass_widget=True)


    # Define the key press callback function
    def key_callback(key):
        # Get the current value of the slider
        current_value = slider_widget.GetRepresentation().GetValue()

        # Update slider value based on key press
        if key == 'Right':
            new_value = min(current_value + 1, all_mem_params.shape[0] - 1)  # Increment with limit
        elif key == 'Left':
            new_value = max(current_value - 1, 0)  # Decrement with limit
        else:
            return

        # Update the slider position
        slider_widget.GetRepresentation().SetValue(new_value)
        create_mesh(new_value, slider_widget)


    # Add the key press callback to the plotter
    plotter.add_key_event("Left", lambda: key_callback("Left"))
    plotter.add_key_event("Right", lambda: key_callback("Right"))


    plotter.show_grid(xlabel='start val', ylabel='middle val', zlabel='right val')
    plotter.show()




