import pickle
import os

plot_folder = f"data_files/visualizations/3D_Plots/cross_session_preds_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
plot_file = f"cross_session_preds_timepoints_session_{session_id}.fig.pickle"
plot_src= os.path.join(plot_folder, plot_file)

with open(plot_src, 'rb') as file: 
    timepoints_sessions_plot = pickle.load(file)

timepoints_sessions_plot.show() 