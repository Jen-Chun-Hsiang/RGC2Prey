def make_file_name(experiment_name, test_ob_folder, test_bg_folder,
                   noise_level=None, fix_disparity_degree=None):

    parts = [
        experiment_name,
        test_ob_folder,
        test_bg_folder
    ]

    # insert disparity tag if provided
    if fix_disparity_degree is not None:
        parts.append(f"disp{fix_disparity_degree}")

    # insert noise tag if provided
    if noise_level is not None:
        parts.append(f"noise{noise_level}")

    # final constant suffix
    parts.append("cricket_location_prediction")

    # join with underscores
    return "_".join(parts)