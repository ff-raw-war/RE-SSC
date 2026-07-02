function C = soft_threshold(qian, hou)

    C = sign(qian)  .* max(abs(qian)-hou, 0 );
end