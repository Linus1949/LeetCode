package solution.demo;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author Linus
 * 使用ConcurrentHashMap + ConcurrentLinkedQueue + ReadWriteLock实现线程安全的LRU
 */
public class MyLruCache<K, V> {
    /**
     * 缓存最大容量
     */
    private final int capacity;

    private ConcurrentHashMap<K, V> cacheMap;
    private ConcurrentLinkedQueue<K> keys;

    /**
     * 读写锁
     */
    private final ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
    private Lock writeLock = readWriteLock.writeLock();
    private Lock readLock = readWriteLock.readLock();

    public MyLruCache(int capacity) {
        if(capacity<0){
            throw new IllegalArgumentException("Illegal capacity: "+capacity);
        }
        this.capacity = capacity;
        cacheMap = new ConcurrentHashMap<>(capacity);
        keys = new ConcurrentLinkedQueue<>();
    }

    /**
     * put方法配合写锁
     */
    public V put(K key, V value){
        //加些锁
        writeLock.lock();
        try {
            //查看key是否已经存在
            if(cacheMap.containsKey(key)){
                moveToTailOfQueue(key);
                //更新cache
                cacheMap.put(key,value);
                return value;
            }
            //是否超出缓存
            if(cacheMap.size()==capacity){
                //移除oldest元素后，再执行插入操作
                removeOldestKey();
            }
            //key不存在，添加进链表尾部
            keys.add(key);
            cacheMap.put(key,value);
            return value;
        }finally {
            writeLock.unlock();
        }
    }
    /**
     * get方法配合读锁
     */
    public V get(K key){
        //加锁
        readLock.lock();
        try{
            //查看是否存在与map
            if(cacheMap.containsKey(key)){
                //重新更新到链表尾部
                moveToTailOfQueue(key);
                return cacheMap.get(key);
            }
            //不存在
            return null;
        }finally {
            readLock.unlock();
        }
    }
    /**
     * remove方法配合写锁
     */
    public V remove(K key){
        writeLock.lock();
        try{
            //判断是否存在于map
            if(cacheMap.containsKey(key)){
                //链表中移除
                keys.remove(key);
                return cacheMap.remove(key);
            }
            //不存在
            return null;
        }finally {
            writeLock.unlock();
        }
    }
    /**
     * 添加/获取元素时，将元素更新到尾部
     */
    private void moveToTailOfQueue(K key){
        keys.remove(key);
        keys.add(key);
    }
    /**
     * 移除队列最老的元素(链表头部),当缓存已满时执行
     */
    private void removeOldestKey(){
        K oldestKey = keys.poll();
        if(oldestKey!=null){
            cacheMap.remove(oldestKey);
            //不需要修改capacity大小，因为配合新元素的插入，会回复到max状态
        }
    }
    /**
     * 返还当前cache的size
     */
    public int size(){
        return cacheMap.size();
    }
}
