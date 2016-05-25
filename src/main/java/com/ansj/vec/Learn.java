package com.ansj.vec;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;

import com.ansj.vec.util.MapCount;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/*
http://blog.csdn.net/itplus/article/details/37969979
 */
public class Learn {
    private static Logger logger = LoggerFactory.getLogger(Learn.class);
    public Map<String, Neuron> wordMap = new HashMap<String,Neuron>();

    private int layerSize = 200;//训练多少个特征
    private int window = 5; //上下文窗口大小 2*window + 1

    private double sample = 1e-3;
    private double alpha = 0.025;
    private double startingAlpha = alpha;

    public int EXP_TABLE_SIZE = 1000;

    private Boolean isCbow = false;

    private double[] expTable = new double[EXP_TABLE_SIZE];

    private int trainWordsCount = 0;

    private int MAX_EXP = 6;

    public Learn(Boolean isCbow, Integer layerSize, Integer window, Double alpha, Double sample) {
            createExpTable();
            if (isCbow != null) {
                this.isCbow = isCbow;
            }
            if (layerSize != null)
                this.layerSize = layerSize;
            if (window != null)
                this.window = window;
            if (alpha != null)
                this.alpha = alpha;
            if (sample != null)
                this.sample = sample;
    }

    public Learn() {
            createExpTable();
    }

    /**
     * trainModel
     * @throws IOException 
     */
    private void trainModel(File file) {
        BufferedReader br = null;
        FileReader fileReader = null;
        InputStreamReader isr  = null;

        try {
            fileReader = new FileReader(file);
            String encode = fileReader.getEncoding();
            isr = new InputStreamReader(new FileInputStream(file), encode);
            br = new BufferedReader(isr);

            String temp = null;
            long nextRandom = 5;
            int wordCount = 0;
            int lastWordCount = 0;
            int wordCountActual = 0;
            while ((temp = br.readLine()) != null) {
                if (wordCount - lastWordCount > 10000) {
                    System.out
                        .println("alpha:" + alpha + "\tProgress: "
                                 + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
                                 + "%");
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                    if (alpha < startingAlpha * 0.0001) {
                        alpha = startingAlpha * 0.0001;
                    }
                }
                String[] strs = temp.split(" ");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; ++i) {
                    Neuron entry = wordMap.get(strs[i]);
                    if (null == entry ) {
                        continue;
                    }
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                     * (sample * trainWordsCount) / entry.freq;
                        nextRandom = nextRandom * 25214903917L + 11;
                        if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                            continue;
                        }
                    }
                    sentence.add((WordNeuron) entry);
                }

                for (int index = 0; index < sentence.size(); ++ index) {
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (isCbow) {
                        cbowGram(index, sentence, (int) nextRandom % window);
                    } else {
                        skipGram(index, sentence, (int) nextRandom % window);
                    }
                }

            }
            System.out.println("Vocab size: " + wordMap.size());
            System.out.println("Words in train file: " + trainWordsCount);
            System.out.println("sucess train over!");
        }catch (FileNotFoundException fnne){
            logger.error( "trainModel() FileNotFoundException! ",fnne);
        }catch( IOException  ioe){
            logger.error( "trainModel() IOException ! ",ioe);
        }catch (Exception e){
            logger.error( "trainModel() error! ",e);
        }finally {
            try {
                fileReader.close();
                br.close();
                isr.close();
            } catch (IOException ioe) {
                logger.error( "trainModel() IOException ! ",ioe);
            }catch (Exception e){
                logger.error("",e);
            }
        }
    }

    /**
     * skip gram 模型训练
     * @param index  词在句子中的索引
     * @param sentence
     * @param b 0-window的一个随机整数，随机缩小语义窗，加快训练速度
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b) {
        // TODO Auto-generated method stub
        try {
            WordNeuron word = sentence.get(index);
            int a, c = 0;
            for (a = b; a < window * 2 + 1 - b; ++a) {
                if (a == window) {
                    continue;
                }
                c = index - window + a;
                if (c < 0 || c >= sentence.size()) {
                    continue;
                }

                double[] neu1e = new double[layerSize];//误差项,每个词建立一次
                //HIERARCHICAL SOFTMAX
                List<Neuron> neurons = word.neurons;
                WordNeuron we = sentence.get(c);
                for (int i = 0; i < neurons.size(); ++ i) {
                    HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                    double f = 0;
                    // Propagate hidden -> output
                    for (int j = 0; j < layerSize; ++j) {
                        f += we.syn0[j] * out.syn1[j];
                    }
                    if (f <= -MAX_EXP || f >= MAX_EXP) {
                        continue;
                    } else {
                        f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                        f = expTable[(int) f];
                    }
                    // 'g' is the gradient multiplied by the learning rate
                    double g = (1 - word.codeArr[i] - f) * alpha;
                    // Propagate errors output -> hidden
                    for (c = 0; c < layerSize; ++c) {
                        neu1e[c] += g * out.syn1[c];
                    }
                    // Learn weights hidden -> output
                    for (c = 0; c < layerSize; ++c) {
                        out.syn1[c] += g * we.syn0[c];
                    }
                }

                // Learn weights input -> hiddenj
                for (int j = 0; j < layerSize; ++ j) {
                    we.syn0[j] += neu1e[j];
                }
            }
        }catch (Exception e){
            logger.error("skipGram() error! ",e);
        }

    }

    /**
     * 词袋模型
     * @param index 词在句子中的索引
     * @param sentence
     * @param b   0-window的一个随机整数，随机缩小语义窗，加快训练速度
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        try {
            WordNeuron word = sentence.get(index);
            int a = 0, c = 0;

            List<Neuron> neurons = word.neurons; //路径上的节点
            double[] neu1e = new double[layerSize];//误差项
            double[] neu1 = new double[layerSize];//误差项，输入窗内的求和向量
            WordNeuron last_word;

            //语义窗内的词向量求和
            for (a = b; a < window * 2 + 1 - b; ++ a) {
                //如果不是自身
                if (a != window) {
                    c = index - window + a;
                    if (c < 0 || c >= sentence.size())
                        continue;
                    last_word = sentence.get(c);
                    if (null == last_word )
                        continue;
                    for (c = 0; c < layerSize; ++ c)
                        neu1[c] += last_word.syn0[c];
                }
            }

            //HIERARCHICAL SOFTMAX
            //优化路径上每个节点
            for (int d = 0; d < neurons.size(); ++ d) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(d); //路径上的一节点
                double f = 0;
                // Propagate hidden -> output
                for (c = 0; c < layerSize; ++ c) {
                    f += neu1[c] * out.syn1[c]; // f = 语义词向量的和 * 该节点的输出向量
                }
                //计算softmax(f)
                if (f <= - MAX_EXP)
                    continue;
                else if (f >= MAX_EXP)
                    continue;
                else {
                    f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                }
                // 'g' is the gradient multiplied by the learning rate
                //            double g = (1 - word.codeArr[d] - f) * alpha;
                //            double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
                double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
                //
                for (c = 0; c < layerSize; ++ c) {
                    neu1e[c] += g * out.syn1[c];
                }
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; ++c) {
                    out.syn1[c] += g * neu1[c]; //out为引用
                }
            }
            for (a = b; a < window * 2 + 1 - b; ++ a) {
                if (a != window) {
                    c = index - window + a;
                    if (c < 0)
                        continue;
                    if (c >= sentence.size())
                        continue;
                    last_word = sentence.get(c);
                    if (null == last_word )
                        continue;
                    for (c = 0; c < layerSize; ++ c)
                        last_word.syn0[c] += neu1e[c];
                }

            }
        }catch(Exception e){
            logger.error("cbowGram() error! ",e );
        }
    }

    /**
     * 统计词频
     * @param file
     * @throws IOException
     */
    private void readVocab(File file) throws IOException {
        MapCount<String> mc = new MapCount<String>();
        BufferedReader br = null;
        FileReader fileReader = null;
        InputStreamReader isr  = null;

        try  {
            fileReader = new FileReader(file);
            String encode = fileReader.getEncoding();
            isr = new InputStreamReader(new FileInputStream(file), encode);
            br = new BufferedReader(isr);

            String temp = null;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split(" ");
                trainWordsCount += split.length;
                for (String string : split) {
                    mc.add(string);
                }
            }
        }catch(FileNotFoundException fnne){
            logger.error("readVocab() FileNotFoundException! ",fnne);
        }catch (IOException ioe){
            logger.error("readVocab() IOException! ",ioe);
        }catch(Exception e){
            logger.error("readVocab() Exception! ",e);
        }finally {
            try {
                fileReader.close();
                br.close();
                isr.close();
            } catch (IOException ioe) {
                logger.error( "trainModel() IOException ! ",ioe);
            }catch (Exception e){
                logger.error("",e);
            }
        }
        for (Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(),
                    (double) element.getValue() / mc.size(), layerSize));
        }
    }

    /**
     * 对文本进行预分类
     *
     * @param files
     * @throws IOException
     * @throws FileNotFoundException
     */
    private void readVocabWithSupervised(File[] files) throws IOException {
        BufferedReader br = null;
        FileReader fileReader = null;
        InputStreamReader isr  = null;

        for (int category = 0; category < files.length; ++ category) {
            // 对多个文件学习
            MapCount<String> mc = new MapCount<String>();
            try{
                fileReader = new FileReader(files[category]);
                String encode = fileReader.getEncoding();
                isr = new InputStreamReader(new FileInputStream(files[category]), encode);
                br = new BufferedReader(isr);


                String temp = null;
                while ((temp = br.readLine()) != null) {
                    String[] split = temp.split(" ");
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                }
            }catch(FileNotFoundException fnne){
                logger.error("FileNotFoundException! ",fnne);
            }catch (IOException ioe){
                logger.error(" IOException! ",ioe);
            }catch(Exception e){
                logger.error(" Exception! ",e);
            }finally {
                try {
                    fileReader.close();
                    br.close();
                    isr.close();
                } catch (IOException ioe) {
                    logger.error( "trainModel() IOException ! ",ioe);
                }catch (Exception e){
                    logger.error("",e);
                }
            }
            for (Entry<String, Integer> element : mc.get().entrySet()) {
                double tarFreq = (double) element.getValue() / mc.size();
                if (wordMap.get(element.getKey()) != null) {
                    double srcFreq = wordMap.get(element.getKey()).freq;
                    if (srcFreq >= tarFreq) {
                        continue;
                    } else {
                        Neuron wordNeuron = wordMap.get(element.getKey());
                        wordNeuron.category = category;
                        wordNeuron.freq = tarFreq;
                    }
                } else {
                    wordMap.put(element.getKey(), new WordNeuron(element.getKey(),
                            (int)tarFreq, category, layerSize));
                }
            }
        }
    }
    /**
     * Precompute the exp() table
     * f(y) = x / (x + 1)
     * x = exp(y)
     *
     * y = MAX_EXP *[ (i * 2) / EXP_TABLE_SIZE   -1  ]
     * 用时，即，知道y，反推i， i = (y + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)
     */
    private void createExpTable() {
        try {
            for (int i = 0; i < EXP_TABLE_SIZE; ++ i) {
                expTable[i] = Math.exp( (i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP );
                expTable[i] = expTable[i] / (expTable[i] + 1);
            }
        }catch (Exception e){
            logger.error("createExpTable() error! ",e);
        }
    }

    /**
     * 根据文件学习
     * @param file
     * @throws IOException 
     */
    public void learnFile(File file) throws IOException {
        try {
            readVocab(file);
            new Haffman(layerSize).make(wordMap.values());

            //查找每个神经元
            for (Neuron neuron : wordMap.values()) {
                ((WordNeuron) neuron).makeNeurons();
            }
            trainModel(file);
        }catch (Exception e){
            logger.error("learnFile() error ! ",e);
        }
    }

    /**
     * 根据预分类的文件学习
     *
     * @param summaryFile
     *          合并文件
     * @param classifiedFiles
     *          分类文件
     * @throws IOException
     */
    public void learnFile(File summaryFile, File[] classifiedFiles) {
        try {
            readVocabWithSupervised(classifiedFiles);
            new Haffman(layerSize).make(wordMap.values());
            // 查找每个神经元
            for (Neuron neuron : wordMap.values()) {
                ((WordNeuron) neuron).makeNeurons();
            }
            trainModel(summaryFile);
        }catch (IOException ioe){
            logger.error("IOE",ioe);
        }catch (Exception e){
            logger.error("",e);
        }
    }
    /**
     * 保存模型
     */
    public void saveModel(File file) {
        // TODO Auto-generated method stub
        try  {
            DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                    new FileOutputStream(file)));
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn0 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
            dataOutputStream.close();
        }catch(FileNotFoundException fnne){
            logger.error("saveModel() FileNotFoundException! ",fnne);
        }catch (IOException ioe){
            logger.error("saveModel() IOException! ",ioe);
        }catch(Exception e){
            logger.error("saveModel() Exception! ",e);
        }
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public double getSample() {
        return sample;
    }

    public void setSample(double sample) {
        this.sample = sample;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public Boolean getIsCbow() {
        return isCbow;
    }

    public void setIsCbow(Boolean isCbow) {
        this.isCbow = isCbow;
    }

    public void setWordMap(Map<String, Neuron> wordMap){
        this.wordMap = wordMap;
    }
    Map<String, Neuron> getWordMap(){
        return this.wordMap;
    }

    public void loadOnlineModel(String modelPath){
        try {
            //this.wordMap = (Map<String, Neuron>) KarlSerializationHelper.read(modelPath);
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
            this.wordMap =  (Map<String, Neuron>) _Deserialization(ois);
            ois.close();

        }catch(FileNotFoundException fnne){
            logger.warn(modelPath + "FileNotFoundException!",fnne);
        } catch (Exception e){
            logger.error("Lode Online Model Occurs ERROR!",e);
        }
    }
    public void saveOnlineModel(String modelPath){
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
            _Serialization(oos);
            oos.close();
        }catch(Exception e){
            logger.error("saveOnlineModel Error!",e);
        }

    }

    private void _Serialization(ObjectOutputStream oos){
        try {
            oos.writeInt(this.wordMap.size());
            for (Entry<String, Neuron> element : this.wordMap.entrySet()) {
                oos.writeUTF(element.getKey());
                oos.writeObject((WordNeuron) element.getValue());
            }
        }catch(Exception e){
            logger.error("序列化失败！",e);
        }
    }
    private Map<String, Neuron>  _Deserialization(ObjectInputStream ois){
        try {
            Map<String, Neuron> d_wordMap = new HashMap<String, Neuron>();
            int num = ois.readInt();
            for (int i = 0; i < num; ++i) {
                String name = ois.readUTF();
                //System.out.println(name);
                WordNeuron obj = (WordNeuron) ois.readObject();
                d_wordMap.put(name, obj);
            }
            return d_wordMap;
        }catch (Exception e){
            logger.error("反序列化失败",e);
            return null;
        }
    }


    public static void main(String[] args) throws IOException {
//        Learn learn = new Learn();
//        long start = System.currentTimeMillis() ;
//        learn.learnFile(new File("library/xh.txt"));
//        System.out.println("use time "+(System.currentTimeMillis()-start));
//        learn.saveModel(new File("library/javaVector"));



    }
}
