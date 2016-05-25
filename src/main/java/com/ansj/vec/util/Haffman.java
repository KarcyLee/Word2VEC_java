package com.ansj.vec.util;

import java.util.Collection;
import java.util.List;
import java.util.TreeSet;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 构建Haffman编码树
 * @author ansj
 *
 */
public class Haffman {
    private static Logger logger = LoggerFactory.getLogger(Haffman.class);
    private int layerSize = 0;

    public Haffman(int layerSize) {
        this.layerSize = layerSize;
    }

    private TreeSet<Neuron> set = new TreeSet<Neuron>();//小顶堆

    public void make(Collection<Neuron> neurons) {
        try {
            set.addAll(neurons);
            while (set.size() > 1) {
                //直到只剩一棵树
                merger();
            }
        }catch (Exception e){
            logger.error("make() error ",e);
        }
    }


    private void merger() {
        // 每次从森林中选取两颗根节点最小的树进行融合
        try {
            HiddenNeuron hn = new HiddenNeuron(layerSize);
            Neuron min1 = set.pollFirst();
            Neuron min2 = set.pollFirst();
            hn.category = min2.category;
            hn.freq = min1.freq + min2.freq;
            min1.parent = hn;
            min2.parent = hn;
            min1.code = 0;//左为0
            min2.code = 1;//右为1
            set.add(hn);
        }catch (Exception e){
            logger.error("merger() error ",e);
        }
    }
    
}
