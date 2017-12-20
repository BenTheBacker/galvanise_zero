from pprint import pprint

import attr

from ggplearn.util import attrutil, func, broker, runprocs


def setup():
    from ggplib.util.init import setup_once
    setup_once()


def test_chunks():
    res = list(func.chunks(range(10), 2))
    assert res[0] == [0, 1]
    assert res[4] == [8, 9]


def test_challenge():
    m = broker.challenge(128)

    print m
    assert len(m) == 128

    m = broker.response(m)

    print m
    assert len(m) == 128


@attr.s
class DummyMsg(object):
    what = attr.ib()


class BrokerX(broker.Broker):
    the_client = None
    got = None

    def on_call_me(self, client, msg):
        self.the_client = client
        self.got = msg.what


def test_broker():
    b = BrokerX()

    b.register(DummyMsg, b.on_call_me)
    m = DummyMsg("hello")
    b.onMessage(42, broker.Message(broker.clz_to_name(DummyMsg), payload=m))

    assert b.the_client == 42
    assert b.got == "hello"


def test_client():
    client = broker.Client(None)
    client.logical_connection = True

    data = client.format_msg(DummyMsg("hello world!"))
    client.rxd.append(data)

    msg = list(client.unbuffer_data())[0]

    assert msg.name == "test_util.DummyMsg"
    assert msg.payload.what == "hello world!"


def test_broker_frag_msg():
    client = broker.Client(None)
    client.logical_connection = True

    data = client.format_msg(DummyMsg("hello world2!"))
    assert len(data) > 14
    client.rxd.append(data[:10])

    empty = list(client.unbuffer_data())
    assert empty == []

    client.rxd.append(data[10:])
    buflens = [len(x) for x in client.rxd]
    assert buflens[0] == 10
    assert buflens[1] > 10

    msg = list(client.unbuffer_data())[0]
    assert msg is not None

    assert msg.name == "test_util.DummyMsg"
    assert msg.payload.what == "hello world2!"


@attr.s
class Container(object):
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()


def test_attrs_recursive():
    print 'test_attrs_recursive.1'

    c = Container(DummyMsg('a'),
                  DummyMsg('b'),
                  DummyMsg('c'))

    m = Container(DummyMsg('o'),
                  DummyMsg('p'),
                  DummyMsg(c))

    d = attrutil.asdict_plus(m)
    pprint(d)

    r = attrutil.fromdict_plus(d)
    assert isinstance(r, Container)

    assert r.x.what == 'o'
    assert r.z.what.x.what == 'a'

    json_str = attrutil.attr_to_json(m, indent=4)
    print json_str

    k = attrutil.json_to_attr(json_str)
    assert k.x.what == 'o'
    assert k.z.what.x.what == 'a'


@attr.s
class Sample(object):
    name = attr.ib()
    data = attr.ib(attr.Factory(list))


@attr.s
class Samples(object):
    k = attr.ib()
    samples = attr.ib(attr.Factory(list))


def test_attrs_listof():
    s0 = Sample('s0', [1, 2, 3, 4, 5])
    s1 = Sample('s1', [5, 4, 3, 2, 1])

    samples = Samples(42, [s0, s1])

    d = attrutil.asdict_plus(samples)
    pprint(d)

    r = attrutil.fromdict_plus(d)

    pprint(r)

    assert isinstance(r, Samples)
    assert len(r.samples) == 2
    assert r.samples[0].name == "s0"
    assert r.samples[1].data[1] == 4


def test_runcmds():
    from twisted.internet import reactor

    def done():
        print "SUCCESS"
        reactor.crash()

    cmds = ["ls -l", "sleep 3", "python2 -c 'import sys; print >>sys.stderr, 123'"]
    run_cmds = runprocs.RunCmds(cmds, cb_on_completion=done)

    reactor.callLater(0.1, run_cmds.spawn)
    reactor.run()


def test_runcmds2():
    from twisted.internet import reactor

    def done():
        print "SUCCESS"
        reactor.crash()

    cmds = ["ls -l", "ls -l"]

    try:
        run_cmds = runprocs.RunCmds(cmds, cb_on_completion=done)
        raise Exception("Should not get here")
    except AssertionError:
        pass

    cmds = ['python -c "import time, signal; signal.signal(signal.SIGTERM, lambda a,b: time.sleep(5)); time.sleep(5)"']
    run_cmds = runprocs.RunCmds(cmds, cb_on_completion=done)

    reactor.callLater(0.1, run_cmds.spawn)
    reactor.run()
