label_lookup = Dict(
    "mnist_train" => "MNIST",
    "emnist_letters_train" => "EMNIST Letters",
    "emnist_digits_train" => "EMNIST Digits",
    "omniglot_train" => "Omniglot",
    "humansketches" => "HumanSketches",
    "opencv"=>"opencv",
    "taco_rle"=>"TACO (RLE)",
    "finch_sparse"=>"Finch (SparseList Walk)",
    "finch_gallop"=>"Finch (SparseList Gallop)",
    "finch_lead"=>"Finch (SparseList Lead)",
    "finch_follow"=>"Finch (SparseList Follow)",
    "finch_vbl"=>"Finch (SparseBlockList)",
    "finch_rle"=>"Finch (RunList)",
    "finch_uint8"=>"Finch (SparseList) (UInt8)",
    "finch_uint8_gallop"=>"Finch (Gallop) (UInt8)",
    "finch_uint8_vbl"=>"Finch (SparseBlockList) (UInt8)",
    "finch_uint8_rle"=>"Finch (RunList) (UInt8)"
)

label(key) = label_lookup[key]