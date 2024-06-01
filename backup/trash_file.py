# PCA with components that explain 90% of variance
required_var_explained = 0.9
explained_var = 0
component_index = 0
component_vars = [explained_var_component for explained_var_component in pca.explained_variance_ratio_]
while explained_var < required_var_explained:
    explained_var += component_vars[component_index]
    component_index += 1
n_components = component_index