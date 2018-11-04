struct Para {
	bool is_color, is_first_frame, scale_adaption, den_per_channel;
	int fixed_area, hog_cell_area, n_bins, scale_num, hog_scale_cell_area, scale_model_max_area;
	double area_norm_factor, inner_padding, lambda, output_sigma_factor, scale_sigma_factor, scale_model_factor, scale_step, scale_learning_rate, cf_learning_rate, hist_learning_rate, merge_factor, init_score, lost_threshold;
	Point2i img_size, target_pos, target_size, target_center, fg_size, bg_size, norm_bg_size, norm_fg_size, cf_size, norm_target_size, norm_delta_size, norm_pwp_size, output_sigma_size;
};
