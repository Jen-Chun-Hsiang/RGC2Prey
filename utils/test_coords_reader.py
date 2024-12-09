from utils.utils import mat_to_dataframe, dataframe_to_dict

mat_file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary.mat'
df = mat_to_dataframe(mat_file_path, summary_key="summary")  # Specify 'custom_variable_name' if different

id_column = "image_id"  # Column to use as the key
value_columns = ["coord_x", "coord_y"]  # Columns to use as the values
result_dict = dataframe_to_dict(df, id_column, value_columns)

img_id = 1
print(f"{img_id}: {result_dict.get(img_id)}")
img_id = 2
print(f"{img_id}: {result_dict.get(img_id)}")