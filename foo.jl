using BenchmarkTools
@inline function unsafe_round_UInt8(x)
    unsafe_trunc(UInt8, round(x))
end
@btime begin
    r = UInt8(255)
    s = UInt8(128)
    for i = 1:10000000
        r = unsafe_round_UInt8(0.5*r+0.5*s);
    end
    println(r)
end