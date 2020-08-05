package solution.demo;

import java.util.LinkedList;

/**
 * 生产者消费者模式：使用Object.wait()/notify()方法实现
 */
public class Storge {
    //载体,
    private LinkedList<Object> list = new LinkedList<>();

    public void produce(){
        synchronized (list){
            //capacity
            int maxSize = 10;
            while(list.size()+1> maxSize){
                System.out.println("生产者"+ Thread.currentThread().getName()
                        + "仓库已满");
                try{
                    list.wait();
                }catch (InterruptedException e){
                    e.printStackTrace();
                }
            }
            list.add(new Object());
            System.out.println("生产者"+ Thread.currentThread().getName()
                    + "生产一个产品，现库存"+ list.size());
            list.notifyAll();
        }
    }

    public void consume(){
        synchronized (list){
            while (list.size()==0){
                System.out.println("消费者"+ Thread.currentThread().getName()
                        + "仓库为空");
                try{
                    list.wait();
                }catch (InterruptedException e){
                    e.printStackTrace();
                }
            }
            list.remove();
            System.out.println("消费者" + Thread.currentThread().getName()
                    + "消费一个产品，现库存" + list.size());
            list.notifyAll();
        }
    }
}

class Produce implements Runnable{
    private Storge storge;

    public Produce(){};

    public Produce(Storge storge, String name){
        this.storge = storge;
    }
    @Override
    public void run() {
        while (true){
            try {
                Thread.sleep(1000);
                storge.produce();
            }catch(InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}

class Consumer implements Runnable{
    private Storge storge;

    public Consumer(){};

    @Override
    public void run() {
        while (true){
            try{
                Thread.sleep(3000);
                storge.consume();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}