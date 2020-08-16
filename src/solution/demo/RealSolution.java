package solution.demo;

import java.util.*;
import java.util.stream.Collectors;

class WorkflowNode {
    String nodeId;
    int timeoutMillis;
    List<WorkflowNode> nextNodes;
    boolean initialised;

    public static WorkflowNode load(String value) {
        // Create head node;
        Map<String, WorkflowNode> map = new HashMap<>();
        WorkflowNode head = new WorkflowNode("HEAD", 0, null);
        map.put(head.nodeId, head);

        for (String nodeValue : value.split("\\|")) {
            String[] properties = nodeValue.split("\\`");
            WorkflowNode node = map.get(properties[0]);

            node.timeoutMillis = Integer.parseInt(properties[1]);
            node.initialised = true;

            // Check next nodes
            if (properties[2].equals("END")) {
                continue;
            }
            node.nextNodes = Arrays.stream(properties[2].split(","))
                    .map(p -> new WorkflowNode(p, 0, null))
                    .collect(Collectors.toList());
            node.nextNodes.forEach(p -> map.put(p.nodeId, p));

            map.put(node.nodeId, node);
        }

        return head;
    }

    public WorkflowNode(String nodeId, int timeoutMillis, List<WorkflowNode> nextNodes) {
        this.nodeId = nodeId;
        this.timeoutMillis = timeoutMillis;
        this.nextNodes = nextNodes;
    }

    public static int maxTime(WorkflowNode head){
        if(!head.initialised){
            return -1;
        }
        int maxTime = 0;
        Stack<WorkflowNode> stack = new Stack<>();
        stack.push(head);
        while (!stack.isEmpty()){
            WorkflowNode temp = stack.pop();
            if(temp.initialised){
                if(temp.nextNodes!=null){
                    for(WorkflowNode node: temp.nextNodes){
                        node.timeoutMillis += temp.timeoutMillis;
                        maxTime = Math.max(maxTime, node.timeoutMillis);
                        stack.push(node);
                    }
                }
            }
            else{
                return -1;
            }
        }
        return maxTime;
    }

    public static void main(String args[])
    {
        Scanner cin = new Scanner(System.in);
        while (cin.hasNext())
        {
            try{
                WorkflowNode node = WorkflowNode.load(cin.next());
                System.out.println(WorkflowNode.maxTime(node));
            }catch (Exception e){
                System.out.println(-1);
            }
        }
    }
}