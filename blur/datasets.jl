download_cache = joinpath(@__DIR__, "../cache")

using ZipFile
using Images
using TestImages
using MLDatasets
using Downloads
using DelimitedFiles
using GZip
using CSV
using DataFrames

function unzip(file,exdir="",flatten=false)
    fileFullPath = isabspath(file) ?  file : joinpath(pwd(),file)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    zarchive = ZipFile.Reader(fileFullPath)
    for f in zarchive.files
        fullFilePath = joinpath(outPath,f.name)
        if flatten 
            if !(endswith(f.name,"/") || endswith(f.name,"\\"))
                write(joinpath(outPath,basename(fullFilePath)), read(f))
            end
        else
            if (endswith(f.name,"/") || endswith(f.name,"\\"))
                mkdir(fullFilePath)
            else
                write(fullFilePath, read(f))
            end
        end
    end
    close(zarchive)
end

function download_dataset(url, name)
    path = joinpath(download_cache, name)
    fname = joinpath(path, basename(url))
    if !isfile(fname)
        mkpath(path)
        Downloads.download(url, fname)
        return fname, true
    else
        return fname, false
    end
end

function download_humansketches()
    sketches_link = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    loc, unpack = download_dataset(sketches_link, "sketches")
    unzip_path = joinpath(dirname(loc), "pngs")
    if unpack
        unzip(loc, unzip_path, true)
    end
    return unzip_path
end

const remotefiles = [
    "airplaneF16.tiff",
    "autumn_leaves.png",
    "barbara_color.png",
    "barbara_gray_512.bmp",
    "bark_512.tiff",
    "bark_he_512.tiff",
    "beach_sand_512.tiff",
    "beach_sand_he_512.tiff",
    "blobs.gif",
    "brick_wall_512.tiff",
    "brick_wall_he_512.tiff",
    "calf_leather_512.tiff",
    "calf_leather_he_512.tiff",
    "cameraman.tif",
    "chelsea.png",
    "coffee.png",
    "earth_apollo17.jpg",
    "fabio_color_256.png",
    "fabio_color_512.png",
    "fabio_gray_256.png",
    "fabio_gray_512.png",
    "grass_512.tiff",
    "grass_he_512.tiff",
    "hela-cells.tif",
    "herringbone_weave_512.tiff",
    "herringbone_weave_he_512.tiff",
    "house.tif",
    "jetplane.tif",
    "lake_color.tif",
    "lake_gray.tif",
    "lena_color_256.tif",
    "lena_color_512.tif",
    "lena_gray_16bit.png",
    "lena_gray_256.tif",
    "lena_gray_512.tif",
    "lighthouse.png",
    "lilly.jpg",
    "livingroom.tif",
    "m51.tif",
    "mandril_color.tif",
    "mandril_gray.tif",
    "mandrill.tiff",
    "monarch_color.png",
    "monarch_color_256.png",
    "moonsurface.tiff",
    "morphology_test_512.tiff",
    "mountainstream.png",
    "mri-stack.tif",
    "multi-channel-time-series.ome.tif",
    "peppers_color.tif",
    "peppers_gray.tif",
    "pigskin_512.tiff",
    "pigskin_he_512.tiff",
    "pirate.tif",
    "plastic_bubbles_512.tiff",
    "plastic_bubbles_he_512.tiff",
    "raffia_512.tiff",
    "raffia_he_512.tiff",
    "resolution_test_1920.tif",
    "resolution_test_512.tif",
    "simple_3d_ball.tif",
    "simple_3d_psf.tif",
    "straw_512.tiff",
    "straw_he_512.tiff",
    "sudoku.tiff",
    "toucan.png",
    "walkbridge.tif",
    "water_512.tiff",
    "water_he_512.tiff",
    "woman_blonde.tif",
    "woman_darkhair.tif",
    "wood_grain_512.tiff",
    "wood_grain_he_512.tiff",
    "woolen_cloth_512.tiff",
    "woolen_cloth_he_512.tiff"
]

const remotefiles_dip3e = [
    "Fig0101(1921 digital image).tif",
    "Fig0102(1922 digital image).tif",
    "Fig0103(1929 digital image Pershing and Foch).tif",
    "Fig0104(Ranger7_1st picture_of_moon).tif",
    "Fig0106(a)(bone-scan-GE).tif",
    "Fig0106(b)(PET_image).tif",
    "Fig0106(c)(cygnusloop-gamma).tif",
    "Fig0106(d)(gamma-Wehe-enhanced).tif",
    "Fig0107(a)(chest-xray-vandy).tif",
    "Fig0107(b)(kidney-original).tif",
    "Fig0107(c)(headCT-Vandy).tif",
    "Fig0107(d)(Cktboard-Lixi).tif",
    "Fig0107(e)(cygnusloop-Xray).tif",
    "Fig0108(a) (corn-fluorescence).tif",
    "Fig0108(b) (smutcorn-fluorescence).tif",
    "Fig0108(c)(cygnusloop-extreme ultraviolet).tif",
    "Fig0109(a)(taxol-anti-cancer agent).tif",
    "Fig0109(b)(cholesterol).tif",
    "Fig0109(c)(microporcessor).tif",
    "Fig0109(d)(nickel oxide thin film).tif",
    "Fig0109(e)(surface of audio CD).tif",
    "Fig0109(f)(organic superconductor).tif",
    "Fig0110(1)(WashingtonDC Band1).TIF",
    "Fig0110(2)(WashingtonDC Band2).TIF",
    "Fig0110(3)(WashingtonDC Band3).TIF",
    "Fig0110(4)(WashingtonDC Band4).TIF",
    "Fig0110(5)(WashingtonDC Band5).TIF",
    "Fig0110(6)(WashingtonDC Band6).TIF",
    "Fig0110(7)(WashingtonDC Band7).TIF",
    "Fig0111(katrina_2005_08_29_NOAA).tif",
    "Fig0112(0)(small-gray-map-americas).tif",
    "Fig0112(1)(top-canada).tif",
    "Fig0112(2)(2nd-from-top-USA).tif",
    "Fig0112(3)(3rd-from-top-Central-Amer).tif",
    "Fig0112(4)(4th-from-top-South-Amer).tif",
    "Fig0113(0)(small-gray-map-europe-russia-etc).tif",
    "Fig0113(1)(left-africa-europe).tif",
    "Fig0113(2)(center-russia).tif",
    "Fig0113(3)(right-South-East-Asia).tif",
    "Fig0113(4)(right-bottom-Australia).tif",
    "Fig0114(a)(cktboard).tif",
    "Fig0114(b)(pills).tif",
    "Fig0114(c)(bottles).tif",
    "Fig0114(d)(bubbles).tif",
    "Fig0114(e)(cereal).tif",
    "Fig0114(f)(lens-Perceptics).tif",
    "Fig0115(a)(thum-print-loop).tif",
    "Fig0115(b)(100-dollars).tif",
    "Fig0115(c)(Korea-Plate).tif",
    "Fig0115(d)(Mexico-Plate).tif",
    "Fig0116(Radar_Tibet_Mountains-highres).tif",
    "Fig0117(a)(MRI-of-knee-Univ-Mich).tif",
    "Fig0117(b)(MRI-spine1-Vandy).tif",
    "Fig0118(a)(crabpulsar-gamma).tif",
    "Fig0118(b)(crabpulsar-xray).tif",
    "Fig0118(c)(crabpulsar-optical).tif",
    "Fig0118(d)(crabpulsar-infrared).tif",
    "Fig0118(e)(crabpulsar-radio).tif",
    "Fig0119(Seismic-Marmousi_img).tif",
    "Fig0120(a)(ultrasound-fetus1).tif",
    "Fig0120(b)(ultrasound-fetus2).tif",
    "Fig0120(c)(ultrasound-tharoid structures).tif",
    "Fig0120(d)(ultrasound-muscle-layers-with-lesion).tif",
    "Fig0121(a)(tungsten_flmt).tif",
    "Fig0121(b)(blown_ic_hr).tif",
    "Fig0122(a)(fractal-iris).tif",
    "Fig0122(b)(fractal-moonscape).tif",
    "Fig0122(c)(skull).tif",
    "Fig0122(d)(lung).tif",
    "Fig0207(a)(gray level band).tif",
    "Fig0219(rose1024).tif",
    "Fig0220(a)(chronometer 3692x2812  2pt25 inch 1250 dpi).tif",
    "Fig0221(a)(ctskull-256).tif",
    "Fig0222(a)(face).tif",
    "Fig0222(b)(cameraman).tif",
    "Fig0222(c)(crowd).tif",
    "Fig0226(galaxy_pair_original).tif",
    "Fig0227(a)(washington_infrared).tif",
    "Fig0228(a)(angiography_mask_image).tif",
    "Fig0228(b)(angiography_live_ image).tif",
    "Fig0229(a)(tungsten_filament_shaded).tif",
    "Fig0229(b)(tungsten_sensor_shading).tif",
    "Fig0230(a)(dental_xray).tif",
    "Fig0230(b)(dental_xray_mask).tif",
    "Fig0232(a)(partial_body_scan).tif",
    "Fig0235(c)(kidney_original).tif",
    "Fig0236(a)(letter_T).tif",
    "Fig0237(a)(characters test pattern)_POST.tif",
    "Fig0240(a)(apollo 17_boulder_noisy).tif",
    "Fig0241(a)(einstein low contrast).tif",
    "Fig0241(b)(einstein med contrast).tif",
    "Fig0241(c)(einstein high contrast).tif",
    "Fig0304(a)(breast_digital_Xray).tif",
    "Fig0305(a)(DFT_no_log).tif",
    "Fig0307(a)(intensity_ramp).tif",
    "Fig0308(a)(fractured_spine).tif",
    "Fig0309(a)(washed_out_aerial_image).tif",
    "Fig0310(b)(washed_out_pollen_image).tif",
    "Fig0312(a)(kidney).tif",
    "Fig0314(a)(100-dollars).tif",
    "Fig0316(1)(top_left).tif",
    "Fig0316(2)(2nd_from_top).tif",
    "Fig0316(3)(third_from_top).tif",
    "Fig0316(4)(bottom_left).tif",
    "Fig0320(1)(top_left).tif",
    "Fig0320(2)(2nd_from_top).tif",
    "Fig0320(3)(third_from_top).tif",
    "Fig0320(4)(bottom_left).tif",
    "Fig0323(a)(mars_moon_phobos).tif",
    "Fig0326(a)(embedded_square_noisy_512).tif",
    "Fig0327(a)(tungsten_original).tif",
    "Fig0333(a)(test_pattern_blurring_orig).tif",
    "Fig0334(a)(hubble-original).tif",
    "Fig0335(a)(ckt_board_saltpep_prob_pt05).tif",
    "Fig0338(a)(blurry_moon).tif",
    "Fig0340(a)(dipxe_text).tif",
    "Fig0342(a)(contact_lens_original).tif",
    "Fig0343(a)(skeleton_orig).tif",
    "Fig0354(a)(einstein_orig).tif",
    "Fig0359(a)(headCT_Vandy).tif",
    "Fig0417(a)(barbara).tif",
    "Fig0418(a)(ray_traced_bottle_original).tif",
    "Fig0419(a)(text_gaps_of_1_and_2_pixels).tif",
    "Fig0421(car_newsprint_sampled_at_75DPI).tif",
    "Fig0422(newspaper_shot_woman).tif",
    "Fig0424(a)(rectangle).tif",
    "Fig0425(a)(translated_rectangle).tif",
    "Fig0427(a)(woman).tif",
    "Fig0429(a)(blown_ic).tif",
    "Fig0431(d)(blown_ic_crop).tif",
    "Fig0432(a)(square_original).tif",
    "Fig0438(a)(bld_600by600).tif",
    "Fig0441(a)(characters_test_pattern).tif",
    "Fig0442(a)(characters_test_pattern).tif",
    "Fig0445(a)(characters_test_pattern).tif",
    "Fig0448(a)(characters_test_pattern).tif",
    "Fig0450(a)(woman_original).tif",
    "Fig0451(a)(satellite_original).tif",
    "Fig0457(a)(thumb_print).tif",
    "Fig0458(a)(blurry_moon).tif",
    "Fig0459(a)(orig_chest_xray).tif",
    "Fig0462(a)(PET_image).tif",
    "Fig0464(a)(car_75DPI_Moire).tif",
    "Fig0465(a)(cassini).tif",
    "Fig0503 (original_pattern).tif",
    "Fig0504(a)(gaussian-noise).tif",
    "Fig0504(b)(rayleigh-noise).tif",
    "Fig0504(c)(gamma-noise).tif",
    "Fig0504(g)(neg-exp-noise).tif",
    "Fig0504(h)(uniform-noise).tif",
    "Fig0504(i)(salt-pepper-noise).tif",
    "Fig0505(a)(applo17_boulder_noisy).tif",
    "Fig0507(a)(ckt-board-orig).tif",
    "Fig0507(b)(ckt-board-gauss-var-400).tif",
    "Fig0508(a)(circuit-board-pepper-prob-pt1).tif",
    "Fig0508(b)(circuit-board-salt-prob-pt1).tif",
    "Fig0510(a)(ckt-board-saltpep-prob.pt05).tif",
    "Fig0512(a)(ckt-uniform-var-800).tif",
    "Fig0512(b)(ckt-uniform-plus-saltpepr-prob-pt1).tif",
    "Fig0513(a)(ckt_gaussian_var_1000_mean_0).tif",
    "Fig0514(a)(ckt_saltpep_prob_pt25).tif",
    "Fig0516(a)(applo17_boulder_noisy).tif",
    "Fig0516(c)(BW_banreject_order4).tif",
    "Fig0519(a)(florida_satellite_original).tif",
    # "Fig0520(a)(NASA_Mariner6_Mars).tif", # This seems to be broken
    "Fig0524(a)(impulse).tif",
    "Fig0524(b)(blurred-impulse).tif",
    "Fig0525(a)(aerial_view_no_turb).tif",
    "Fig0525(b)(aerial_view_turb_c_0pt0025).tif",
    "Fig0525(c)(aerial_view_turb_c_0pt001).tif",
    "Fig0525(d)(aerial_view_turb_c_0pt00025).tif",
    "Fig0526(a)(original_DIP).tif",
    "Fig0529(a)(noisiest_var_pt1).tif",
    "Fig0529(d)(medium_noise_var_pt01).tif",
    "Fig0529(g)(least_noise_var_10minus37).tif",
    "Fig0533(a)(circle).tif",
    "Fig0534(a)(ellipse_and_circle).tif",
    "Fig0539(a)(vertical_rectangle).tif",
    "Fig0539(c)(shepp-logan_phantom).tif",
    "Fig0608(RGB-full-color-cube).tif",
    "Fig0620(a)(picker_phantom).tif",
    "Fig0621(a)(weld-original).tif",
    "Fig0622(a)(tropical_rain_grayscale.tif",
    "Fig0624 (garments).tif",
    "Fig0627(a)(WashingtonDC Band3-RED).TIF",
    "Fig0627(b)(WashingtonDC Band2-GREEN).TIF",
    "Fig0627(c)(1)(WashingtonDC Band1-BLUE).TIF",
    "Fig0627(d)(WashingtonDC Band4).TIF",
    "Fig0628(a)(jupiter-moon.-Io).tif",
    "Fig0628(b)(jupiter-Io-closeup).tif",
    "Fig0630(01)(strawberries_fullcolor).tif",
    "Fig0631(a)(strawberries_coffee_full_color).tif",
    "Fig0635(bottom_left_stream).tif",
    "Fig0635(middle_row_left_chalk ).tif",
    "Fig0635(top_ left_flower).tif",
    "Fig0636(woman_baby_original).tif",
    "Fig0637(a)(caster_stand_original).tif",
    "Fig0638(a)(lenna_RGB).tif",
    "Fig0642(a)(jupiter_moon_original).tif",
    "Fig0645(a)(RGB1-red).tif",
    "Fig0645(b)(RGB1-green).tif",
    "Fig0645(c)(RGB1-blue).tif",
    "Fig0645(e)(RGB2_red).tif",
    "Fig0645(f)(RGB2_green).tif",
    "Fig0645(g)(RGB2_blue).tif",
    "Fig0646(a)(lenna_original_RGB).tif",
    "Fig0648(a)(lenna-noise-R-gauss-mean0-var800).tif",
    "Fig0648(b)(lenna-noise-G-gauss-mean0-var800).tif",
    "Fig0648(c)(lenna-noise-B-gauss-mean0-var800).tif",
    "Fig0650(a)(rgb_image_G_saltpep_pt05).tif",
    "Fig0651(a)(flower_no_compression).tif",
    "Fig0701.tif",
    "Fig0723(a).tif",
    "Fig0726(a).tif",
    "Fig0734(a).tif",
    "Fig0801(a).tif",
    "Fig0801(b).tif",
    "Fig0801(c).tif",
    "Fig0809(a).tif",
    "Fig0816(a).tif",
    "Fig0819(a).tif",
    "Fig0834(a).tif",
    "Fig0835(a).tif",
    "Fig0837(a).tif",
    "Fig0840_0021.tif",
    "Fig0840_0044.tif",
    "Fig0840_0201.tif",
    "Fig0840_0266.tif",
    "Fig0840_0424.tif",
    "Fig0840_0801.tif",
    "Fig0840_0959.tif",
    "Fig0840_1023.tif",
    "Fig0840_1088.tif",
    "Fig0840_1224.tif",
    "Fig0840_1303.tif",
    "Fig0840_1304.tif",
    "Fig0840_1595.tif",
    "Fig0840_1609.tif",
    "Fig0840_1652.tif",
    "Fig0850(a).tif",
    "Fig0905(a)(wirebond-mask).tif",
    "Fig0907(a)(text_gaps_1_and_2_pixels).tif",
    "Fig0911(a)(noisy_fingerprint).tif",
    "Fig0914(a)(licoln from penny).tif",
    "Fig0916(a)(region-filling-reflections).tif",
    "Fig0918(a)(Chickenfilet with bones).tif",
    "Fig0929(a)(text_image).tif",
    "Fig0931(a)(text_image).tif",
    "Fig0935(a)(ckt_board_section).tif",
    "Fig0937(a)(ckt_board_section).tif",
    "Fig0938(a)(cygnusloop_Xray_original).tif",
    "Fig0939(a)(headCT-Vandy).tif",
    "Fig0940(a)(rice_image_with_intensity_gradient).tif",
    "Fig0941(a)(wood_dowels).tif",
    "Fig0943(a)(dark_blobs_on_light_background).tif",
    "Fig0944(a)(calculator).tif",
    "Fig1001(a)(constant_gray_region).tif",
    "Fig1001(b)(edge_image).tif",
    "Fig1001(c)(thresholded_image).tif",
    "Fig1001(d)(noisy_region).tif",
    "Fig1001(e)(edge_noisy_image).tif",
    "Fig1001(f)(region_split_merge_image).tif",
    "Fig1004(b)(turbine_blade_black_dot).tif",
    "Fig1005(a)(wirebond_mask).tif",
    "Fig1007(a)(wirebond_mask).tif",
    "Fig1008(a)(step edge).tif",
    "Fig1008(b)(ramp edge).tif",
    "Fig1008(c)(roof_edge).tif",
    "Fig1016(a)(building_original).tif",
    "Fig1022(a)(building_original).tif",
    "Fig1025(a)(building_original).tif",
    "Fig1026(a)(headCT-Vandy).tif",
    "Fig1027(a)(van_original).tif",
    "Fig1030(a)(tooth).tif",
    "Fig1034(a)(marion_airport).tif",
    "Fig1036(a)(original_septagon).tif",
    "Fig1036(b)(gaussian_noise_mean_0_std_10_added).tif",
    "Fig1036(c)(gaussian_noise_mean_0_std_50_added).tif",
    "Fig1037(a)(septagon_gaussian_noise_mean_0_std_10_added).tif",
    "Fig1037(b)(intensity_ramp).tif",
    "Fig1038(a)(noisy_fingerprint).tif",
    "Fig1039(a)(polymersomes).tif",
    "Fig1040(a)(large_septagon_gaussian_noise_mean_0_std_50_added).tif",
    "Fig1041(a)(septagon_small_noisy_mean_0_stdv_10).tif",
    "Fig1042(a)(septagon_small_noisy_mean_0_stdv_10).tif",
    "Fig1043(a)(yeast_USC).tif",
    "Fig1045(a)(iceberg).tif",
    "Fig1046(a)(septagon_noisy_shaded).tif",
    "Fig1048(a)(yeast_USC).tif",
    "Fig1049(a)(spot_shaded_text_image).tif",
    "Fig1050(a)(sine_shaded_text_image).tif",
    "Fig1051(a)(defective_weld).tif",
    "Fig1053(a)(cygnusloop_Xray_original).tif",
    "Fig1056(a)(blob_original).tif",
    "Fig1057(a)(small_blobs-original).tif",
    "Fig1059(a)(AbsADI).tif",
    "Fig1059(b)(PosADI).tif",
    "Fig1059(c)(NegADI).tif",
    "Fig1060(a)(car on left).tif",
    "Fig1060(b)(car on right).tif",
    "Fig1060(c)(car removed).tif",
    "Fig1061(LANDSAT_with moving target).tif",
    "Fig1105(a)(noisy_stroke).tif",
    "Fig1108(a)(mapleleaf).tif",
    "Fig1111(a)(triangle).tif",
    "Fig1111(b)(square).tif",
    "Fig1116(leg_bone).tif",
    "Fig1120(a)(chromosome_boundary).tif",
    "Fig1122(1)(top-canada).tif",
    "Fig1122(2)(2nd-from-top-USA).tif",
    "Fig1122(3)(3rd-from-top-Central-Amer).tif",
    "Fig1122(4)(4th-from-top-South-Amer).tif",
    "Fig1127(a)(WashingtonDC Band4).tif",
    "Fig1128(a)(superconductor-smooth-texture)-DO NOT SEND.tif",
    "Fig1128(b)(cholesterol-rough-texture)-DO NOT SEND.tif",
    "Fig1128(c)(microporcessor-regular texture)-DO NOT SEND.tif",
    "Fig1130(a)(uniform_noise).tif",
    "Fig1130(b)(sinusoidal).tif",
    "Fig1130(c)(cktboard_section).tif",
    "Fig1135(a)(random_matches).tif",
    "Fig1135(b)(ordered_matches).tif",
    "Fig1137(a)(painting_original_padded).tif",
    "Fig1137(b)(painting_translated_padded).tif",
    "Fig1137(c)(painting_halfsize_padded).tif",
    "Fig1137(d)(painting_mirrored_padded).tif",
    "Fig1137(e)(painting_rot45deg).tif",
    "Fig1137(f)(painting_rot90deg_padded).tif",
    "Fig1138(a)(WashingtonDC_Band1_564).tif",
    "Fig1138(b)(WashingtonDC_Band2_564).tif",
    "Fig1138(c)(WashingtonDC_Band3_564).tif",
    "Fig1138(d)(WashingtonDC_Band4_564).tif",
    "Fig1138(e)(WashingtonDC_Band5_564).tif",
    "Fig1138(f)(WashingtonDC_Band6_564).tif",
    "Fig1204(WashingtonDC ).tif",
    "Fig1209(a)(Hurricane Andrew).tif",
    "Fig1209(b)(eye template).tif",
    "Fig1213(a)(WashingtonDC_Band1_512).tif",
    "Fig1213(b)(WashingtonDC_Band2_512).tif",
    "Fig1213(c)(WashingtonDC_Band3_512).tif",
    "Fig1213(d)(WashingtonDC_Band4_512).tif",
    "Fig1213(e)(Mask_B1_without_numbers).tif",
    "Fig1213(e)(Mask_B2_without_numbers).tif",
    "Fig1213(e)(Mask_B3_without_numbers).tif",
    "Fig1213(e)(Mask_Composite_without_numbers).tif",
    "Fig1218(airplanes).tif",
    "Fig1225(keys).tif",
    "FigP0211.tif",
    "FigP0215.tif",
    "FigP0223.tif",
    "FigP0302(a).tif",
    "FigP0302(b).tif",
    "FigP0302(c).tif",
    "FigP0311.tif",
    "FigP0314(a).tif",
    "FigP0314(b).tif",
    "FigP0321(a).tif",
    "FigP0321(b).tif",
    "FigP0321(c).tif",
    "FigP0332.tif",
    "FigP0407.tif",
    "FigP0421(a).tif",
    "FigP0421(b).tif",
    "FigP0421(left)(padded_image).tif",
    "FigP0421(right)(center-padded_image).tif",
    "FigP0422(a).tif",
    "FigP0422(b).tif",
    "FigP0433(a).tif",
    "FigP0433(b).tif",
    "FigP0433(left)(DIP_image).tif",
    "FigP0436(a).tif",
    "FigP0436(b).tif",
    "FigP0436(left)(hand_xray).tif",
    "FigP0438(a).tif",
    "FigP0438(b).tif",
    "FigP0438(c).tif",
    "FigP0438(d).tif",
    "FigP0438(left).tif",
    "FigP0501(filtering).tif",
    "FigP0501.tif",
    "FigP0510(a).tif",
    "FigP0510(b).tif",
    "FigP0510(left).tif",
    "FigP0510(right).tif",
    "FigP0520(blurred-heart).tif",
    "FigP0520.tif",
    "FigP0528(a)(single_dot).tif",
    "FigP0528(a).tif",
    "FigP0528(b)(two_dots).tif",
    "FigP0528(b).tif",
    "FigP0528(c)(doughnut).tif",
    "FigP0528(c).tif",
    "FigP0605.tif",
    "FigP0606(color_bars).tif",
    "FigP0606.tif",
    "FigP0615.tif",
    "FigP0616(a).tif",
    "FigP0616(b).tif",
    "FigP0616(c).tif",
    "FigP0625.tif",
    "FigP0717.tif",
    "FigP0724(left).tif",
    "FigP0724(original).tif",
    "FigP0724.tif",
    "FigP0725.tif",
    "FigP0726.tif",
    "FigP0905(U).tif",
    "FigP0905(a).tif",
    "FigP0905(b).tif",
    "FigP0905(c).tif",
    "FigP0905(d).tif",
    "FigP0905(top).tif",
    "FigP0906.tif",
    "FigP0917(noisy_rectangle).tif",
    "FigP0918(a).tif",
    "FigP0918(b).tif",
    "FigP0918(c).tif",
    "FigP0918(left).tif",
    "FigP0919(UTK).tif",
    "FigP0919(a).tif",
    "FigP0919(b).tif",
    "FigP0920(a).tif",
    "FigP0920(b).tif",
    "FigP0920(c).tif",
    "FigP0934(blobs_in_circular_arrangement).tif",
    "FigP0934.tif",
    "FigP0936(bubbles_on_black_background).tif",
    "FigP0936.tif",
    "FigP0937.tif",
    "FigP1007(a).tif",
    "FigP1007(b).tif",
    "FigP1012.tif",
    "FigP1036(blobs).tif",
    "FigP1036.tif",
    "FigP1039.tif",
    "FigP1043.tif",
    "FigP1109.tif",
    "FigP1110.tif",
    "FigP1111.tif",
    "FigP1126(bottles).tif",
    "FigP1126.tif",
    "FigP1127(bubbles).tif",
    "FigP1127.tif",
    "FigP1206.tif",
    "Figp0917.tif",
]

function list_files_without_extensions(directory)
    filenames = readdir(directory)
    filenames_without_extensions = [splitext(filename)[1] for filename in filenames]
    return filenames_without_extensions
end

dip3e_masks = list_files_without_extensions(joinpath(@__DIR__, "dip3e_masks"))

"""
humansketches(idx)

Return a grascale image A[vertical pixel position, horizontal pixel position],
measured from image upper left. Pixel values are stored using 8-bit grayscale
values. `idx` is specifies which sketch image to load. The sketches number from
1:20_000.
"""
function humansketches(idx)
    @boundscheck begin
        idx in 1:20_000 || throw(BoundsError("humansketches", idx))
    end

    path = download_humansketches()

    Array{Gray{N0f8}, 2}(load(joinpath(path, "$idx.png")))
end

"""
`mnist_train(i::Int)`

Return a single image from the MNIST train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function mnist_train(i)
    dir = joinpath(download_cache, "mnist")
    return MNIST(:train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`fashionmnist_train(i::Int)`

Return a single image from the FashionMNIST train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function fashionmnist_train(i)
    dir = joinpath(download_cache, "fashionmnist")
    return FashionMNIST(:train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`omniglot_train(i::Int)`

Return a single image from the Omniglot train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function omniglot_train(i)
    dir = joinpath(download_cache, "omniglot")
    return Omniglot(:train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`mnist_test(i::Int)`

Return a single image from the MNIST test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function mnist_test(i)
    dir = joinpath(download_cache, "mnist")
    return MNIST(:test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`fashionmnist_test(i::Int)`

Return a single image from the FashionMNIST test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function fashionmnist_test(i)
    dir = joinpath(download_cache, "fashionmnist")
    return FashionMNIST(:test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`omniglot_test(i::Int)`

Return a single image from the Omniglot test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function omniglot_test(i)
    dir = joinpath(download_cache, "omniglot")
    return Omniglot(:test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`omniglot_small1(i::Int)`

Return a single image from the Omniglot small1 dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function omniglot_small1(i)
    dir = joinpath(download_cache, "omniglot")
    return Omniglot(:small1, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`omniglot_small2(i::Int)`

Return a single image from the Omniglot small2 dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function omniglot_small2(i)
    dir = joinpath(download_cache, "omniglot")
    return Omniglot(:small2, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_digits_test(i::Int)`

Return a single image from the EMNIST digits test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_digits_test(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:digits, :test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_digits_train(i::Int)`

Return a single image from the EMNIST digits train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_digits_train(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:digits, :train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_letters_test(i::Int)`

Return a single image from the EMNIST letters test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_letters_test(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:letters, :test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_letters_train(i::Int)`

Return a single image from the EMNIST letters train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_letters_train(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:letters, :train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_test(i::Int)`

Return a single image from the complete EMNIST test dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_test(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:byclass, :test, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`emnist_train(i::Int)`

Return a single image from the complete EMNIST train dataset. The image is selected based on the integer input `i`, representing the index of the image. Pixel values are 8-bit grayscale.
"""
function emnist_train(i)
    dir = joinpath(download_cache, "emnist")
    return EMNIST(:byclass, :train, dir=dir, Tx=UInt8).features[:, :, i]
end

"""
`willow_gen(n::Int)`

Generates a pattern based on the input `n` and returns it as a single image. The pattern consists of pixels where the value is determined by a function of the pixel's position and the input `i`, producing a characteristic shape.
"""
function willow_gen(n)
    return [UInt8((i - n/2)^2 + (j - n/2)^2 < (n/4)^2) for i in 1:n, j in 1:n]
end