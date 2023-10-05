label_lookup = Dict(
    "mnist_train" => "MNIST",
    "emnist_letters_train" => "EMNIST Letters",
    "emnist_digits_train" => "EMNIST Digits",
    "omniglot_train" => "Omniglot",
    "humansketches" => "HumanSketches",
    "opencv"=>"OpenCV",
    "taco_rle"=>"TACO (RLE)",
    "finch_sparse"=>"Finch (Sparse)",
    "finch_gallop"=>"Finch (Sparse, Gallop)",
    "finch_lead"=>"Finch (Sparse, Lead)",
    "finch_follow"=>"Finch (Sparse, Follow)",
    "finch_vbl"=>"Finch (Sparse, VBL)",
    "finch_rle"=>"Finch (RLE)",
    "finch_uint8"=>"Finch (Sparse) (UInt8)",
    "finch_uint8_gallop"=>"Finch (Gallop) (UInt8)",
    "finch_uint8_vbl"=>"Finch (VBL) (UInt8)",
    "finch_uint8_rle"=>"Finch (RLE) (UInt8)"
)

label(key) = label_lookup[key]