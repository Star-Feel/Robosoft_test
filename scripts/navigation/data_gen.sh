echo "Running main_data_gen_random_go.py"
python scripts/navigation/main_data_gen_random_go.py

# choiceable, if you want to make the road be short
# python scripts/navigation/main_data_gen_prune.py

echo "Running main_data_gen_obstacle.py"
python scripts/navigation/main_data_gen_obstacle.py

echo "Running main_data_gen_target.py"
python scripts/navigation/main_data_gen_target.py

echo "Running main_data_gen_full.py"
python scripts/navigation/main_data_gen_full.py

echo "Running main_data_gen_meshes.py"
python scripts/navigation/main_data_gen_meshes.py

echo "Running main_data_gen_visual.py"
python scripts/navigation/main_data_gen_visual.py

# python scripts/navigation/main_data_gen_release.py
