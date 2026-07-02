function C = soft_threshold1(qian, hou)

    C = max(qian-hou, 0) + min(qian+hou, 0);
end