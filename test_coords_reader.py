from utils.utils import load_mat_to_dataframe

mat_file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary.mat'
df = load_mat_to_dataframe(mat_file_path)  # Specify 'custom_variable_name' if different
ind_name = 'image_id'
data_dict = df.set_index(ind_name).to_dict(orient='index')

img_id = 1
print(f"{img_id}: {data_dict[img_id]}")
img_id = 2
print(f"{img_id} x: {data_dict[img_id]['coord_x']}")