package solution.demo;

/**
 * 单例模式
 */
public class Singleton {
    /**
     * 饿汉式
     * 优点：实现简单，安全可靠
     * 缺点：在还不需要此实例的时候就把实例创建好了，没有起到Lazy Loading的效果，而且如果有些类一直都没有被
     * 调用过的话会很浪费
     */
    public static class SingletonOne{
        //实例直接在初始化时就创建
        private static SingletonOne instance = new SingletonOne();
        //私有化构造函数
        private SingletonOne(){}

        public static SingletonOne getInstance(){
            return instance;
        }
    }
    /**
     * 懒汉式，初始版本
     * 优点：只有当需要这个类的实例时，再去创建它
     * 缺点：存在线程安全问题
     */
    public static class SingletonLazy{
        private static SingletonLazy instance;
        private SingletonLazy(){};
        public static SingletonLazy getInstance(){
            if(instance==null){
                instance = new SingletonLazy();
            }
            return instance;
        }
    }
    /**
     * 饿汉式，双重校验
     * 通过加锁，可以保证同时只有一个线程走到第二个判断空代码去
     * 使用volatile修饰singleton是为了防止指令重排
     * 如果没有volatile防止指令重排，有很小的概率会出现构造函数被执行多次
     *         memory = allocate(); //1.分配对象内存空间
     *         instance(memory);   //2.初始化对象
     *         instance = memory;  //设置instance执行刚分配的内存地址，此时instance!=NULL
     *     步骤2和步骤3不存在数据依赖关系，而且无论重排前还是重排后程序的执行结果在单线程中并没有改变，
     *     因此这种重排优化是允许的，如果3步骤提前于步骤2，但是instance还没有初始化完成,
     *     但是指令重排只会保证串行语义的执行的一致性（单线程），但并不关心多线程间的语义一致性。
     *     所以当一条线程访问instance不为null时，由于instance示例未必已初始化完成，也就造成了线程安全问题。
     */
    public static class SingletonLazySafe{
        private static volatile SingletonLazySafe singleton;

        private SingletonLazySafe(){};

        public static SingletonLazySafe getInstance(){
            if(singleton==null){
                synchronized (SingletonLazySafe.class){
                    if(singleton == null){
                        singleton = new SingletonLazySafe();
                    }
                }
            }
            return singleton;
        }
    }
    /**
     * 静态内部类，静态内部类不会在Singleton类加载时就加载，而是在调用getInstance()方法时才进行加载，
     * 达到懒加载效果，但存在反射攻击或者反序列化攻击
     * 反射攻击：通过反射获取构造函数，然后通过setAccessible(true)就可以调用私有构造函数
     * 解决：可以修改构造器在被要求创建第二个实例的时候抛出异常, RuntimeException
     * 反序列化攻击：获得单例对象后，通过序列化把对象写入到文件中，然后再读取出来，通过反序列化读取对象
     * 解决：再构造器中抛出对象异常流，ObjectStreamException
     */
    public static class SingletonStatic{
        private static class SingletonHolder{
            private static SingletonStatic instance = new SingletonStatic();
        }
        private SingletonStatic(){}

        public static SingletonStatic getInstance(){
            return SingletonHolder.instance;
        }
    }
    /**
     * 通过枚举实现单例模式
     * 枚举的特性：在编译时会生成一个enum类，类中的每一个元素都会被修饰成public static final
     */
    public enum SingletonEnum{
        INSTANCE;
        public void doSomething(){
            System.out.println("doSomething!");
        }
    }
    //调用时直接使用Singleton.INSTANCE.doSomething()即可
}
