import os

# --- 1. Define the variables you want to save ---
player_name = "Alex"
score = 1250
level = 5
inventory = ["sword", "shield", "health potion"]

# --- 2. Specify the folder and file name ---
save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/temp/'
    
file_name = "test_save_file.txt"

# --- 3. Create the folder if it doesn't exist ---
# The os.makedirs() function will create a directory.
# The `exist_ok=True` argument prevents an error if the folder already exists.
print(f"Checking for '{save_folder}' directory...")
os.makedirs(save_folder, exist_ok=True)
print("Directory is ready.")

# --- 4. Combine folder and file name to create the full path ---
# This ensures the file is saved inside the correct folder.
file_path = os.path.join(save_folder, file_name)

# --- 5. Write the variables to the file ---
# 'w' mode means we are writing to the file. If it already exists, it will be overwritten.
# Using a 'with' block ensures the file is automatically closed even if errors occur.
print(f"Saving data to '{file_path}'...")
with open(file_path, 'w') as f:
    f.write(f"PlayerName:{player_name}\n")
    f.write(f"Score:{score}\n")
    f.write(f"Level:{level}\n")
    # For the list, we'll join the items with a comma
    f.write(f"Inventory:{','.join(inventory)}\n")

print("Save complete!")
