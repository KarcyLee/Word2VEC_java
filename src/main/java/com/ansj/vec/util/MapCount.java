//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ansj.vec.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

public class MapCount<T> {
    private static Logger logger = LoggerFactory.getLogger(MapCount.class);
    private HashMap<T, Integer> hm = null;

    public MapCount() {
        this.hm = new HashMap<T, Integer>();
    }

    public MapCount(int initialCapacity) {
        this.hm = new HashMap<T, Integer>(initialCapacity);
    }

    public void add(T t, int n) {
        try {
            if(! this.hm.containsKey(t)){
                this.hm.put(t, Integer.valueOf(n));
            } else {
                this.hm.put(t, Integer.valueOf(n));
            }
        }catch (Exception e){
            logger.error("add() error ",e);
        }

    }

    public void add(T t) {
        try {
            this.add(t, 1);
        }catch (Exception e){
            logger.error("add() error ",e);
        }
    }

    public int size() {
        return this.hm.size();
    }

    public void remove(T t) {
        this.hm.remove(t);
    }

    public HashMap<T, Integer> get() {
        return this.hm;
    }

    public String getDic() {
        try {
            Iterator iterator = this.hm.entrySet().iterator();
            StringBuilder sb = new StringBuilder();
            Entry next = null;

            while (iterator.hasNext()) {
                next = (Entry) iterator.next();
                sb.append(next.getKey());
                sb.append("\t");
                sb.append(next.getValue());
                sb.append("\n");
            }

            return sb.toString();
        }catch (Exception e){
            logger.error("getDic() error ",e);
            return "getDic() error!";
        }
    }

    public static void main(String[] args) {
        System.out.println(9223372036854775807L);
    }
}
