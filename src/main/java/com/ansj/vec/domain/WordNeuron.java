package com.ansj.vec.domain;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class WordNeuron extends Neuron {
    private static final long serialVersionUID = 3788759655867351164L;
    private static Logger logger = LoggerFactory.getLogger(WordEntry.class);
    public String name;
    public double[] syn0 = null; //input->hidden
    public List<Neuron> neurons = null;//路径神经元，均为hiddenNeuron
    public int[] codeArr = null;

    public List<Neuron> makeNeurons() {
        try {
            if (neurons != null) {
                return neurons;
            }
            Neuron neuron = this;
            neurons = new LinkedList<Neuron>();
            while ((neuron = neuron.parent) != null) {
                neurons.add(neuron);
            }
            Collections.reverse(neurons);
            codeArr = new int[neurons.size()];

            for (int i = 1; i < neurons.size(); ++ i) {
                codeArr[i - 1] = neurons.get(i).code;
            }
            codeArr[codeArr.length - 1] = this.code;

            return neurons;
        }catch (Exception e){
            logger.error("makeNeurons() error ",e);
            return null;
        }
    }

    public WordNeuron(String name, double freq, int layerSize) {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        Random random = new Random();
        for (int i = 0; i < syn0.length; i++) {
            syn0[i] = (random.nextDouble() - 0.5) / layerSize;
        }
    }
    /**
     * 用于有监督的创造huffman tree
     * @param name
     * @param freq
     * @param layerSize
     */
    public WordNeuron(String name, double freq, int category, int layerSize) {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        this.category = category;
        Random random = new Random();
        for (int i = 0; i < syn0.length; i++) {
            syn0[i] = (random.nextDouble() - 0.5) / layerSize;
        }
    }


    //序列化
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        out.writeUTF(name);
    }

    //反序列化
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        name = (String) in.readUTF();
    }

}