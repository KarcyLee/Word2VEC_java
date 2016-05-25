package com.ansj.vec;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordEntry;
import com.ansj.vec.domain.WordNeuron;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Word2VEC {
	private static Logger logger = LoggerFactory.getLogger(Word2VEC.class);
	public static void main(String[] args) throws IOException {

		// Learn learn = new Learn();
		// learn.learnFile(new File("library/xh.txt"));
		// learn.saveModel(new File("library/javaSkip1"));

		Word2VEC vec = new Word2VEC();
		vec.loadJavaModel("library/javaSkip1");


		String str = "毛泽东";
		long start = System.currentTimeMillis();
		for (int i = 0; i < 100; i++) {
			System.out.println(vec.distance(str));
			;
		}
		System.out.println(System.currentTimeMillis() - start);

		System.out.println(System.currentTimeMillis() - start);
		// System.out.println(vec2.distance(str));
		//
		//
		// //男人 国王 女人
		// System.out.println(vec.analogy("邓小平", "毛泽东思想", "毛泽东"));
		// System.out.println(vec2.analogy("毛泽东", "毛泽东思想", "邓小平"));
	}

	private HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	private int words;
	private int size; //词向量长度
	private int topNSize = 40;

	/**
	 * 加载增量训练模型
	 * @param path
	 * @throws IOException
     */
	public void loadOnlineModel(String path)throws IOException{
		try {
			Learn learn = new Learn();
			learn.loadOnlineModel(path);
			double[] syn0 = null;
			float[] syn1 = null;
			for (Entry<String, Neuron> element : learn.wordMap.entrySet()) {
				syn0 = ((WordNeuron) element.getValue()).syn0;
				syn1 = new float[syn0.length];
				for (int i = 0; i < syn0.length; ++i) {
					syn1[i] = (float) syn0[i];
				}
				this.wordMap.put(element.getKey(), syn1);
			}
			words = wordMap.size();
			size = syn0.length;
		}catch (Exception e){
			logger.error("LoadOnlineModel Error!",e);
		}

	}


	/**
	 * 加载模型
	 * 
	 * @param path
	 *            模型的路径
	 * @throws IOException
	 */
	public void loadGoogleModel(String path) throws IOException {
		try {
			DataInputStream dis = null;
			BufferedInputStream bis = null;
			double len = 0;
			float vector = 0;
			try {
				bis = new BufferedInputStream(new FileInputStream(path));
				dis = new DataInputStream(bis);
				// //读取词数
				words = Integer.parseInt(readString(dis));
				// //大小
				size = Integer.parseInt(readString(dis));
				String word;
				float[] vectors = null;
				for (int i = 0; i < words; i++) {
					word = readString(dis);
					vectors = new float[size];
					len = 0;
					for (int j = 0; j < size; j++) {
						vector = readFloat(dis);
						len += vector * vector;
						vectors[j] = (float) vector;
					}
					len = Math.sqrt(len);

					for (int j = 0; j < size; j++) {
						vectors[j] /= len;
					}

					wordMap.put(word, vectors);
					dis.read();
				}
			} finally {
				bis.close();
				dis.close();
			}
		}catch (Exception e){
			logger.error("loadGoogleModel() Error! ",e);
		}
	}

	/**
	 * 加载模型
	 * 
	 * @param path
	 *            模型的路径
	 * @throws IOException
	 */
	//public void loadJavaModel(String path){
	public boolean loadJavaModel(String path){
		boolean result = false;
		try {
			DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
			words = dis.readInt();
			size = dis.readInt();

			float vector = 0;

			String key = null;
			float[] value = null;
			for (int i = 0; i < words; i++) {
				double len = 0;
				key = dis.readUTF();
				value = new float[size];
				for (int j = 0; j < size; j++) {
					vector = dis.readFloat();
					len += vector * vector;
					value[j] = vector;
				}

				len = Math.sqrt(len);

				for (int j = 0; j < size; j++) {
					value[j] /= len;
				}
				wordMap.put(key, value);
			}
			dis.close();
			result = true;
		} catch (FileNotFoundException fnfe) {
			logger.error("loadJavaModel() FileNotFoundException! ",fnfe);
			return false;
		} catch (IOException ioe) {
			logger.error("loadJavaModel() IOException! ",ioe);
			return false;
		} catch (Exception e) {
			logger.error("loadJavaModel() Exception! ",e);
			return false;
		}
		return result;
	}

	private static final int MAX_SIZE = 50;

	/**
	 * 近义词
	 * 
	 * @return
	 */
	public TreeSet<WordEntry> analogy(String word0, String word1, String word2) {
		try {
			float[] wv0 = getWordVector(word0);
			float[] wv1 = getWordVector(word1);
			float[] wv2 = getWordVector(word2);

			if (wv1 == null || wv2 == null || wv0 == null) {
				return null;
			}
			float[] wordVector = new float[size];
			for (int i = 0; i < size; i++) {
				wordVector[i] = wv1[i] - wv0[i] + wv2[i];
			}
			float[] tempVector;
			String name;
			List<WordEntry> wordEntrys = new ArrayList<WordEntry>(topNSize);
			for (Entry<String, float[]> entry : wordMap.entrySet()) {
				name = entry.getKey();
				if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
					continue;
				}
				float dist = 0;
				tempVector = entry.getValue();
				for (int i = 0; i < wordVector.length; i++) {
					dist += wordVector[i] * tempVector[i];
				}
				insertTopN(name, dist, wordEntrys);
			}
			return new TreeSet<WordEntry>(wordEntrys);
		}catch (Exception e){
			logger.error("analogy() error! ",e);
			return null;
		}
	}

	private void insertTopN(String name, float score, List<WordEntry> wordsEntrys) {
		// TODO Auto-generated method stub
		try {
			if (wordsEntrys.size() < topNSize) {
				wordsEntrys.add(new WordEntry(name, score));
				return;
			}
			float min = Float.MAX_VALUE;
			int minOffe = 0;
			for (int i = 0; i < topNSize; i++) {
				WordEntry wordEntry = wordsEntrys.get(i);
				if (min > wordEntry.score) {
					min = wordEntry.score;
					minOffe = i;
				}
			}

			if (score > min) {
				wordsEntrys.set(minOffe, new WordEntry(name, score));
			}
		}catch(Exception e){
			logger.error("insertTopN() error! ",e);
		}

	}

	/**
	 * 计算余弦距离，这个词必须是集合里必须有的。
	 *
	 * @return 距离最近的topNSize个词
	 */
	public Set<WordEntry> distance(String queryWord) {
		try {
			float[] center = wordMap.get(queryWord);
			if (center == null) {
				return Collections.emptySet();
			}

			int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
			TreeSet<WordEntry> result = new TreeSet<WordEntry>();

			double min = Float.MIN_VALUE;
			for (Map.Entry<String, float[]> entry : wordMap.entrySet()) {
				float[] vector = entry.getValue();
				float dist = 0;
				for (int i = 0; i < vector.length; i++) {
					dist += center[i] * vector[i];
				}

				if (dist > min) {
					result.add(new WordEntry(entry.getKey(), dist));
					if (resultSize < result.size()) {
						result.pollLast();
					}
					min = result.last().score;
				}
			}
			result.pollFirst();

			return result;
		}catch(Exception e){
			logger.error("distance() error! ",e);
			return null;
		}
	}

	public Set<WordEntry> distance(List<String> words) {
		try {
			float[] center = null;
			for (String word : words) {
				center = sum(center, wordMap.get(word));
			}

			if (center == null) {
				return Collections.emptySet();
			}

			int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
			TreeSet<WordEntry> result = new TreeSet<WordEntry>();

			double min = Float.MIN_VALUE;
			for (Map.Entry<String, float[]> entry : wordMap.entrySet()) {
				float[] vector = entry.getValue();
				float dist = 0;
				for (int i = 0; i < vector.length; i++) {
					dist += center[i] * vector[i];
				}

				if (dist > min) {
					result.add(new WordEntry(entry.getKey(), dist));
					if (resultSize < result.size()) {
						result.pollLast();
					}
					min = result.last().score;
				}
			}
			result.pollFirst();

			return result;
		}catch(Exception e){
			logger.error("distance() error! ",e);
			return null;
		}
	}

	private float[] sum(float[] center, float[] fs) {
		// TODO Auto-generated method stub
		try {
			if (center == null && fs == null) {
				return null;
			}

			if (fs == null) {
				return center;
			}

			if (center == null) {
				return fs;
			}

			for (int i = 0; i < fs.length; i++) {
				center[i] += fs[i];
			}

			return center;
		}catch(Exception e){
			logger.error("sum() error! ",e);
			return null;
		}
	}

	/**
	 * 得到词向量
	 * 
	 * @param word
	 * @return
	 */
	public float[] getWordVector(String word) {
		try {
			return wordMap.get(word);
		}catch(Exception e){
			logger.error("getWordVector() error! ",e);
			return null;
		}
	}

	public static float readFloat(InputStream is) throws IOException {
		try {
			byte[] bytes = new byte[4];
			is.read(bytes);
			return getFloat(bytes);
		}catch (Exception e){
			logger.error("readFloat() error! ",e);
			return 0;
		}
	}

	/**
	 * 读取一个float
	 * 
	 * @param b
	 * @return
	 */
	public static float getFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}

	/**
	 * 读取一个字符串
	 * 
	 * @param dis
	 * @return
	 * @throws IOException
	 */
	private static String readString(DataInputStream dis) throws IOException {
		// TODO Auto-generated method stub
		byte[] bytes = new byte[MAX_SIZE];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) {
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[MAX_SIZE];
			}
		}
		sb.append(new String(bytes, 0, i + 1));
		return sb.toString();
	}

	public int getTopNSize() {
		return topNSize;
	}

	public void setTopNSize(int topNSize) {
		this.topNSize = topNSize;
	}

	public HashMap<String, float[]> getWordMap() {
		return wordMap;
	}

	public int getWords() {
		return words;
	}

	public int getSize() {
		return size;
	}



}
