label_lookup = Dict(
    "mnist_train" => "MNIST",
    "emnist_letters_train" => "EMNIST Letters",
    "emnist_digits_train" => "EMNIST Digits",
    "omniglot_train" => "Omniglot",
    "humansketches" => "HumanSketches",
    "opencv"=>"opencv",
    "taco_rle"=>"TACO (RLE)",
    "finch_sparse"=>"Finch (Sparse)",
    "finch_gallop"=>"Finch (Gallop)",
    "finch_lead"=>"Finch (Lead)",
    "finch_follow"=>"Finch (Follow)",
    "finch_vbl"=>"Finch (VBL)",
    "finch_rle"=>"Finch (RLE)",
    "finch_uint8"=>"Finch (Sparse) (UInt8)",
    "finch_uint8_gallop"=>"Finch (Gallop) (UInt8)",
    "finch_uint8_vbl"=>"Finch (VBL) (UInt8)",
    "finch_uint8_rle"=>"Finch (RLE) (UInt8)"
)

label(key) = label_lookup[key]