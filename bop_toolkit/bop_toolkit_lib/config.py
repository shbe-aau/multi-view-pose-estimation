# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Configuration of the BOP Toolkit."""

######## Basic ########

# Folder with the BOP datasets.
datasets_path = r'/media/shbe/data/share-to-docker/bop/bop-tless-dataset'

# Folder with pose results to be evaluated.
results_path = r'/media/shbe/data/share-to-docker/bop/bop_sample_results/bop_challenge_2019'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/media/shbe/data/share-to-docker/bop/eval'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/media/shbe/data/share-to-docker/bop/output'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
