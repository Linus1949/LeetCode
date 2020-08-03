package solution.demo;

import java.util.*;

public class DijkStra {

    private static final Long MAX_RANGE=0x7fffffffL;

    private int size;

    private List<List<Integer>> graph;
    private List<List<Integer>> weight;

    public DijkStra(List<List<Integer>> graph, List<List<Integer>> weight){
        size = graph.size();
        this.graph = graph;
        this.weight = weight;
    }

    private List<Long> distance;
    private List<Boolean> unchanged;
    private PriorityQueue<DijkstraNode> minHeap;
    public long run(int s, int e) {
        /**
         * initialize
         */
        minHeap = new PriorityQueue<DijkstraNode>(size, new Comparator<DijkstraNode>() {
            @Override
            public int compare(DijkstraNode o1, DijkstraNode o2) {
                return (int)(o1.val - o2.val);
            }
        });
        distance = new ArrayList<Long>(size);
        for (int i = 0; i < size; ++i) {
            distance.add(i, MAX_RANGE);
        }
        unchanged = new ArrayList<Boolean>(size);
        for (int i = 0; i < size; ++i) {
            unchanged.add(i, false);
        }
        /**
         * get started
         */
        minHeap.add(new DijkstraNode(0, s));
        distance.set(s, 0L);
        while (!minHeap.isEmpty()) {
            DijkstraNode top = minHeap.poll();
            int now = top.id;
            if (unchanged.get(top.id)) {
                continue;
            }
            unchanged.set(top.id, true);
            int tot = graph.get(now).size();
            for (int i = 0; i < tot; ++i) {
                int to = graph.get(now).get(i);
                int wei = weight.get(now).get(i);
                if (unchanged.get(to)) {
                    continue;
                }
                if (distance.get(to) >= distance.get(now) + wei) {
                    distance.set(to, distance.get(now) + wei);
                    minHeap.add(new DijkstraNode(distance.get(to), to));
                }
            }
        }
        return distance.get(e);
    }
    public static void main(String[] args) {
        List<List<Integer>> graph = Arrays.asList(Arrays.asList(1,2), Arrays.asList(0,2,3), Arrays.asList(0,1,4), Arrays.asList(1,4), Arrays.asList(2,3));
        List<List<Integer>> weight = Arrays.asList(Arrays.asList(6,4), Arrays.asList(6,1,2), Arrays.asList(4,1,10), Arrays.asList(2,6), Arrays.asList(10,6));
        DijkStra dijkStra = new DijkStra(graph, weight);
        System.out.println(dijkStra.run(0,4));
    }

    class DijkstraNode{
        public long val;
        public int id;
        public DijkstraNode(long val, int id) {
            this.val = val;
            this.id = id;
        }
    }
}
