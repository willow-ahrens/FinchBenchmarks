C = ex.body.lhs.tns.tns
C_val = C.val
A = (ex.body.rhs.args[1]).tns.tns
B = (ex.body.rhs.args[2]).tns.tns
A_stop = (size(A))[1]
B_stop = (size(B))[1]
i_stop = A_stop
C_val = 0.0
A_p = 1
A_i0 = 1
A_i1 = A.idx[A_p]
i = 1
i_start = i
phase_start = max(i_start)
phase_stop = min(B.start - 1, A.idx[end], i_stop)
if phase_stop >= phase_start
    i = i
    i = phase_stop + 1
end
i_start = i
phase_start_2 = max(i_start)
phase_stop_2 = min(B.stop, A.idx[end], i_stop)
if phase_stop_2 >= phase_start_2
    i_2 = i
    i = phase_start_2
    A_p = searchsortedfirst(A.idx, phase_start_2, A_p, length(A.idx), Base.Forward)
    A_i0 = phase_start_2
    A_i1 = A.idx[A_p]
    while i <= phase_stop_2
        i_start_2 = i
        phase_stop_3 = min(A_i1, phase_stop_2)
        i_3 = i
        if A_i1 == phase_stop_3
            i_4 = phase_stop_3
            C_val = C_val + A.val[A_p] * B.val[(i_4 - B.start) + 1]
            A_p += 1
            A_i0 = A_i1 + 1
            A_i1 = A.idx[A_p]
        else
        end
        i = phase_stop_3 + 1
    end
    i = phase_stop_2 + 1
end
i_start = i
phase_start_4 = max(i_start)
phase_stop_4 = min(A.idx[end], i_stop)
if phase_stop_4 >= phase_start_4
    i_5 = i
    i = phase_stop_4 + 1
end
i_start = i
phase_start_5 = max(i_start)
phase_stop_5 = min(B.start - 1, i_stop)
if phase_stop_5 >= phase_start_5
    i_6 = i
    i = phase_stop_5 + 1
end
i_start = i
phase_start_6 = max(i_start)
phase_stop_6 = min(B.stop, i_stop)
if phase_stop_6 >= phase_start_6
    i_7 = i
    i = phase_stop_6 + 1
end
i_start = i
phase_start_7 = max(i_start)
phase_stop_7 = min(i_stop)
if phase_stop_7 >= phase_start_7
    i_8 = i
    i = phase_stop_7 + 1
end
(C = (Scalar){0.0, Float64}(C_val),)