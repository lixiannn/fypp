import os, time
import zmq
import zmq_helper
import json, joblib
import ast
import training, inference
import tensorflow as tf

class Node:
    """A peer-to-peer node that can act as client or server at each round"""

    def __init__(self, context, node_id, peers, cluster_id, role_in_cluster):
        self.context = context
        self.node_id = node_id
        self.peers = peers
        self.cluster_id = cluster_id
        self.role_in_cluster = role_in_cluster
        self.log_prefix = "[" + str(node_id).upper() + "] "
        self.in_connection = None
        self.out_connection = {}
        self.local_model = None  # local model
        self.local_history = None
        self.global_model = None
        self.aggregate_weights = []
        self.initialize_node()

    def initialize_node(self):
        """Creates one zmq.ROUTER socket as incoming connection and n number
        of zmq.DEALER sockets as outgoing connections as per adjacency matrix"""
        self.local_history = []
        self.in_connection = zmq_helper.act_as_server(self.context, self.node_id)
        if len(self.peers) > 0:
            for server_id in self.peers :
                self.out_connection[server_id] = zmq_helper.act_as_client(self.context, server_id, self.node_id)

    def print_node_details(self):
        print("*" * 60)
        print("%s Node ID = %s" % (self.log_prefix, self.node_id))
        print("%s Peer IDs = %s %s" % (self.log_prefix, type(self.peers), self.peers))
        print("%s Cluster ID = %s" % (self.log_prefix, self.cluster_id))
        print("%s Role in Cluster = %s" % (self.log_prefix, self.role_in_cluster))
        print("%s Context = %s" % (self.log_prefix, self.context))
        print("%s Incoming Connection = %s" % (self.log_prefix, self.in_connection))
        print("%s Outgoing Connections = %s" % (self.log_prefix, self.out_connection))
        print("%s History = %s" % (self.log_prefix, self.local_history))
        print("*" * 60)

    def save_model(self):
        model_filename = "/usr/thisdocker/dataset/" + str(self.node_id) + ".pkl"
        joblib.dump(self.local_model, model_filename)

    def load_prev_model(self):
        model_filename = "/usr/thisdocker/dataset/" + str(self.node_id) + ".pkl"
        self.local_model = joblib.load(model_filename)

    def send_model(self, to_node):
        try:
            zmq_helper.send_zipped_pickle(self.out_connection[to_node], self.local_model)
            # self.out_connection[to_node].send_string(self.local_model)
        except Exception as e:
            print("%sERROR establishing socket for to-node" % self.log_prefix)

    def send_weights(self, to_node):
        try:
            zmq_helper.send_zipped_pickle(self.out_connection[to_node], self.aggregate_weights)
            # self.out_connection[to_node].send_string(self.local_model)
        except Exception as e:
            print("%sERROR establishing socket for to-node" % self.log_prefix)

    def send_model_weights(self, to_node):
        try:
            zmq_helper.send_zipped_pickle_mulipart(self.out_connection[to_node], self.global_model, self.aggregate_weights)
            # self.out_connection[to_node].send_string(self.local_model)
        except Exception as e:
            print("%sERROR establishing socket for to-node" % self.log_prefix)
            print("send model weights exception:")
            print(e)

    def receive_weights(self):
        from_node = self.in_connection.recv(0)  # Reads identity
        self.aggregate_weights = zmq_helper.recv_zipped_pickle(self.in_connection)  # Reads weight
        # self.local_model = self.in_connection.recv_string()
        return from_node

    def receive_model(self):
        from_node = self.in_connection.recv(0)  # Reads identity
        self.local_model = zmq_helper.recv_zipped_pickle(self.in_connection)  # Reads model object
        # self.local_model = self.in_connection.recv_string()
        return from_node

    def receive_model_weights(self):
        from_node = self.in_connection.recv(0)  # Reads identity
        self.local_model, self.aggregate_weights = zmq_helper.recv_zipped_pickle_multipart(self.in_connection)  # Reads model object
        self.global_model = self.local_model
        # self.local_model = self.in_connection.recv_string()
        return from_node

    def training_step(self, step):
        # local model training
        build_flag = True if step == 1 else False
        self.local_model = training.local_training(self.node_id, self.local_model, build_flag)
        if build_flag:
            self.global_model = self.local_model
        self.aggregate_weights.append(self.local_model.get_weights())
        # self.local_model = {"from": self.node_id}  # for debugging
        # self.save_model()

    def inference_step(self):
        print("inferencing...")
        inference.eval_on_test_set(self.global_model)

    def fed_average(self):
        print("Fed Averaging")
        print("aggregate_weights: ", self.aggregate_weights)
        avg_weight = []
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*self.aggregate_weights):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_weight.append(layer_mean)
        
        self.global_model.set_weights(avg_weight) 

def main():
    """main function"""
    context = zmq.Context()  # We should only have 1 context which creates any number of sockets
    node_id = os.environ["ORIGIN"]
    peers_list = ast.literal_eval(os.environ["PEERS"])
    cluster_id = os.environ["CLUSTER_ID"]
    role_in_cluster = os.environ["ROLE_IN_CLUSTER"]
    this_node = Node(context, node_id, peers_list, cluster_id, role_in_cluster)


    # Read comm template config file
    comm_template = json.load(open('comm_template.json'))
    total_rounds = len(comm_template.keys())
    print(f"node number: {this_node.node_id} , role: {this_node.role_in_cluster}")
    isLeader = False
    if this_node.role_in_cluster == "LEADER":
        isLeader = True
    print(f"{this_node.node_id} is leader: {isLeader}")

    for i in range(1, total_rounds + 2):
        if i == total_rounds+1:
            if this_node.node_id == "node"+str(comm_template[str(total_rounds)]["to"]):
                # Last node 
                # fed avg
                if this_node.role_in_cluster == "LEADER":
                    print("entering fed avg")
                    this_node.fed_average()
                # Global accuracy
                this_node.inference_step()
        else :
            ith_round = comm_template[str(i)]
            from_node = "node" + str(ith_round["from"])
            to_node = "node" + str(ith_round["to"])
            if node_id == from_node:
                # This node is dealer and receiving node is router
                training_start = time.process_time()
                this_node.training_step(i)
                print("%sTime : Training Step = %s" % (this_node.log_prefix, str(time.process_time() - training_start)))

                print("%sSending iteration %s from %s to %s" % (this_node.log_prefix, str(i), from_node, to_node))
                this_node.send_model_weights(to_node)
                # this_node.send_weights(to_node)
            elif node_id == to_node:
                # This node is router and sending node is dealer
                rcvd_from = this_node.receive_model_weights()
                # rcvd_from = this_node.receive_weights()
                # print("%sReceived object %s at iteration %s" % (this_node.log_prefix, str(this_node.local_model), str(i)))
                print("%sReceived weight %s at iteration %s" % (this_node.log_prefix, str(this_node.aggregate_weights), str(i)))

                this_node.save_model()

                # Logging iteration and prev_node for audit
                this_node.local_history.append({"iteration":i, "prev_node":rcvd_from.decode("utf-8")})

    # this_node.print_node_details()
    



if __name__ == "__main__":
    main()
