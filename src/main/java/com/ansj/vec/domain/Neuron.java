package com.ansj.vec.domain;

import java.io.Serializable;

public abstract class Neuron implements Comparable<Neuron> , Serializable {
    private static final long serialVersionUID = -368849976087750176L;
    public double freq;
    public Neuron parent;
    public int code;
    // 语料预分类
    public int category = -1;
    
    @Override
    public int compareTo(Neuron neuron) {
        // 实现大于号 ，正序排序
        if (this.category == neuron.category) {
            if (this.freq > neuron.freq) {
                return 1;
            } else {
                return -1;
            }
        } else if (this.category > neuron.category) {
            return 1;
        } else {
            return 0;
        }

    }


}
