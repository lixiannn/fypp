import zmq
import pickle, zlib


def act_as_server(context, server_identity):
    """current node connection is configured as server and return zmq socket for all communication"""
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt_string(zmq.IDENTITY, server_identity)
    socket.bind("tcp://*:5555")
    return socket


def act_as_client(context, server_identity, client_identity):
    """current node connection is configured as client, attached to this round's server
     and return zmq socket for all communication"""
    socket = context.socket(zmq.DEALER)
    socket.setsockopt_string(zmq.IDENTITY, client_identity)
    server_string = "tcp://"+str(server_identity)+":5555"
    socket.connect(server_string)
    return socket


def send_zipped_pickle(socket, obj, flags=0, protocol=pickle.HIGHEST_PROTOCOL):
    """pickle the object, compress the pickle and send it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def recv_zipped_pickle(socket, flags=0):
    """reverse compress and pickle operations to get object"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


def send_updated_model(socket, obj, identity, protocol=pickle.HIGHEST_PROTOCOL):
    """used for large model files which may be sent in multiple parts"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send_multipart([identity, z])

def send_zipped_pickle_mulipart(socket, obj, weights, travel_cost, flags=0, protocol=pickle.HIGHEST_PROTOCOL):
    """pickle the object, compress the pickle and send it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    w = pickle.dumps(weights, protocol)
    tc = pickle.dumps(travel_cost, protocol)
    return socket.send_multipart([z, w, tc], flags=flags)


def recv_zipped_pickle_multipart(socket, flags=0):
    """reverse compress and pickle operations to get object"""
    items_received = socket.recv_multipart(flags)
    if len(items_received) == 3:
        p = zlib.decompress(items_received[0])
        weights = pickle.loads(items_received[1])
        travel_cost = pickle.loads(items_received[2])
    model = pickle.loads(p)
    return model, weights, travel_cost