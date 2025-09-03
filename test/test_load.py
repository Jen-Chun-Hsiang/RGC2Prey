import os

# --- 1. Specify the folder and file name to load from ---
save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/temp/'
file_name = "test_save_file.txt"
file_path = os.path.join(save_folder, file_name)

# --- 2. Check if the save file exists before trying to read it ---
if os.path.exists(file_path):
    print(f"Found save file! Loading data from '{file_path}'...")

    # --- 3. Initialize variables to store the loaded data ---
    # It's good practice to give them default values.
    loaded_player_name = ""
    loaded_score = 0
    loaded_level = 0
    loaded_inventory = []

    # --- 4. Read the data from the file ---
    # 'r' mode means we are reading the file.
    with open(file_path, 'r') as f:
        # We read the file line by line
        for line in f:
            # The .strip() method removes any leading/trailing whitespace, including the newline character \n
            # The .split(':', 1) method splits the string only on the FIRST colon it finds.
            key, value = line.strip().split(':', 1)

            # --- 5. Check the key and assign the value to the correct variable ---
            if key == "PlayerName":
                loaded_player_name = value
            elif key == "Score":
                loaded_score = int(value)  # Convert the string '1250' back to an integer
            elif key == "Level":
                loaded_level = int(value)  # Convert the string '5' back to an integer
            elif key == "Inventory":
                # Convert the "sword,shield,health potion" string back into a list
                loaded_inventory = value.split(',')

    # --- 6. Print the loaded data to confirm it worked ---
    print("\n--- Data Loaded Successfully ---")
    print(f"Player Name: {loaded_player_name}")
    print(f"Score: {loaded_score}")
    print(f"Level: {loaded_level}")
    print(f"Inventory: {loaded_inventory}")
    print("---------------------------------")

else:
    # This message will be displayed if the script can't find the save file.
    print(f"ERROR: No save file found at '{file_path}'")
    print("Please run the 'save_script.py' first to create a save file.")
