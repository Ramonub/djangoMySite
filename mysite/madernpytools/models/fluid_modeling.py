import numpy as np
import sympy as symp
import madernpytools.backbone as mbb
import madernpytools.tools.symbolic_tools as msym

g = 9.81  # Gravity constant


class FluidUnitConverter(mbb.IUnitConverter):

    @staticmethod
    def convert(value, unit):
        """ Convert value of unit to unit which is compatible with the Madern Fluid Modeling conventions
        """
        # Make value to match FluidModel (meters and Pascal)
        if 'mm' in unit:
            return value * 1e-3  # Convert mm to m
        elif 'mu' in unit:
            return value * 1e-6  # Convert mu to m
        elif 'kPa' in unit:
            return value * 1e3  # Convert kPa to Pa
        else:
            return value


def f_Re(rho, mu, V, d):
    ''' Returns Reynolds number for circular ducts

    :param rho: [kg/m3] Fluid density
    :param mu: [] Fluid viscosity
    :param V: [m/s] Fluid velocity
    :param d: [d] Pipe diameter
    :return: [-] Reynold's number
    '''
    return rho*V*d/mu


class Fluid(mbb.MadernObject, mbb.XMLSerializer):
    def __init__(self, rho, mu):
        """ Fluid
        :param rho: [kg/m3] Fluid density
        :param mu : [kg/(m*s)] Fluid viscocity
        """

        # Define parameters:
        self.rho = rho
        self.mu = mu

        # Initialize serialization
        mbb.XMLSerializer.__init__(self)

    def __eq__(self, other):
        """ Returns true if self properties match other properties
        """
        return ((self.rho == other.rho) and
                (self.mu == other.mu))

    def __str__(self):
        """ Returns a string with fluid properties
        """
        return 'rho: {0} [kg/m3], mu: {1} [(N*s)/m2]'.format(self.rho, self.mu)


# Define common fluid:
air = Fluid(rho=1.2, mu=1.8e-5)


class Pipe(mbb.MadernObject, mbb.XMLSerializer):

    def __init__(self, d: float, L: float, eps: float, K: float = 0.0):
        """
        :param d  : [m] Diameter
        :param L  : [m] Length
        :param dz : [m] Elevation
        :param eps: [m] Roughness
        :param K  : [m] Minor losses
        """
        self.d = d
        self.L = L
        self.eps = eps
        self.K = K

        # Initialize serialization
        mbb.XMLSerializer.__init__(self)

    @property
    def A(self):
        """
        Returns pipe cross-section [m2]
        """
        return 0.25*np.pi *self. d**2

    def __str__(self):
        return 'd: {0}, L: {1}, eps: {2}'.format(self.d, self.L, self.eps)


class Point(mbb.MadernObject, mbb.XMLSerializer):

    def __init__(self, pressure, z, fluid, name=''):
        """ Fluid evaluation point
        :param pressure: [Pa] Fluid pressure at point
        :param z       : [m] Point height
        :param fluid   : Fluid object
        :param name    : Point name (optional for identification)
        """

        # Define properties:
        self._fluid = fluid
        self._pressure = pressure
        self._z = z
        self._name = name

        # Initialize XML serialization
        mbb.XMLSerializer.__init__(self)

        self._symbol_head = symp.symbols('h_{0}'.format(id(self)))

    @property
    def name(self):
        """Returns point name"""
        return self._name

    @property
    def symbolic_head(self):
        """Returns unique symbol which can be used to represent point head"""
        return  self._symbol_head

    @property
    def fluid(self):
        """Fluid present at point"""
        return self._fluid

    @property
    def z(self):
        """Height of point[m]
        """
        return self._z

    @property
    def pressure(self):
        """Pressure of point [Pa]"""
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        """Pressure of point [Pa]"""
        self._pressure = value

    @property
    def head(self):
        """ Returns point head [m]
        """
        return self. z + self.pressure / (self.fluid.rho*g)

    def __eq__(self, other):
        ''' Verify if self is equal to other

        :param other:
        :return:
        '''

        return ((self.fluid == other.fluid) and
                 (self.pressure == other.pressure) and
                 (self.z == other.z) and
                 (self.name == other.name)
                )

    def __add__(self, other):
        """Returns a new point in which pressure and height are summed"""
        # Check fluid:
        if self.fluid == other.fluid:
            return Point(pressure=self.pressure + other.pressure,
                         z=self.z + other.z,
                         fluid=self.fluid)
        else:
            raise ValueError('Fluids do not match ({0} and {1})'.format(self.fluid, other.fluid))

    def __sub__(self, other):
        """Returns a new point in which pressure and height are substracted (self - other)"""
        if self.fluid == other.fluid:
            return Point(pressure=self.pressure - other.pressure,
                         z=self.z - other.z,
                         fluid=self.fluid)
        else:
            raise ValueError('Fluids do not match ({0} and {1})'.format(self.fluid, other.fluid))

    def __str__(self):
        """ Returns string with point properties
        """
        return "point {3}: z={0}[m], pressure={1:.2f} [kPa], fluid={2}".format(self.z, self.pressure * 1e-3, self.fluid,
                                                                               self.name)


class PipeFlow(object):

    def __init__(self, pipe, p_in: Point, p_out: Point):
        """ Represents the flow in a pipe between point p_in and p_out.
        """
        self._pipe = pipe
        self._points = {'in': p_in, 'out': p_out}
        self._V = 1.0

        # Update flow velocity
        self._updateV()

    @property
    def points(self):
        """Returns points in which flow exists"""
        return self._points

    @property
    def head_loss(self):
        """ Flow  head loss [m]
        """
        return (self._points['out'] - self._points['in']).head  # + self.minor_head_loss

    def _updateV(self):
        """ Update th flow velocity
        """
        if abs(self._V) < 1e-6:
            # Ensure iteration is activated:
            print('Setting velocity\n dh:{0:.3f}\nV: {1:.4e}'.format(self.head_loss, self._V))
            self._V = 1.0 * np.sign(self.head_loss)

        dV = 1.0
        cnt = 0
        while abs(dV) > 1e-4:
            f = self.f  # Compute f (based on current V)
            if f > 0:
                dh = self.head_loss  # Head loss
                sign = np.sign(dh)  # get head sign to maintain flow direction
                v_new = sign * PipeFlow.compute_flow_velocity(self._pipe, f, abs(dh))  # Compute velocity
                dV = self._V - v_new  # update dV for convergence check
                self._V = v_new  # Update internal state
            else:
                # No friction, as a result of no flow
                self._V = 0
                dV = 0  # quit loop
            cnt += 1

            if cnt > 100:
                raise RuntimeError('Did not converge dV: {2}\n dh:{0:.3f}\nV: {1:.4e},\nf: {3}'.format(
                    self.head_loss, self._V, dV, f))

    @staticmethod
    def compute_flow_velocity(pipe, f, dh):
        """ Compute flow velocity based on pipe, friction and head-loss

        Implementation according to eq. 6.10 of 8th edition Fluid Mechanics by White

        :param pipe: Pipe object in which flow exists
        :param f:  Darcy friction factor for the flow
        :param dh:  Head-loss
        :return: Flow velocity
        """

        if abs(f * pipe.L / pipe.d + pipe.K) < 1e-6:
            print('Returned zero velocity')
            raise RuntimeError('pipe: {0}\n f: {1:.4f}\n dh: {2}'.format(pipe, f, dh))
            return 0
        else:
            friction_loss = f * pipe.L / pipe.d + pipe.K
            return (2 * dh * g * friction_loss ** -1) ** 0.5

    @property
    def V(self):
        """ Flow speed [m/s]
        """
        self._updateV()
        return self._V

    @property
    def Q(self):
        """
        Returns pipe flow [m3/s]
        """
        V = self.V
        if V is not None:
            return self._pipe.A * V
        else:
            return None

    @property
    def Qhour(self):
        """
        Returns pipe flow [m3/s]
        """
        V = self.V
        if V is not None:
            return self._pipe.A * V * 3600
        else:
            return None

    @staticmethod
    def _f_update(fluid, pipe, V, f_est):
        """ Returns Darcy's friction factor based on estimate f_est

        :param fluid:
        :param pipe:
        :param V:
        :param f_est:
        :return:
        """
        return (-2.0 * np.log10(2.51 / (f_Re(fluid.rho, fluid.mu, V, pipe.d) * f_est ** 0.5)  # [-]
                                + ((pipe.eps / pipe.d) / 3.7))
                ) ** -2

    @property
    def f(self):
        """
        Darcy friction factor for current pipe flow
        Computed iteratively using Colebrook's formula (see eq. 6.48 of 8th edition Fluid Mechanics by White)
        :return:
        """

        # Compute friction factor
        if abs(self._V) > 1e-4:
            f_est = 0.001
            dest = 1
            # Iterate friction factor
            cnt = 0
            while dest > 1e-7:
                f_new = PipeFlow._f_update(fluid=self.points['in'].fluid,
                                           pipe=self._pipe,
                                           V=abs(self._V),
                                           f_est=f_est)
                dest = abs(f_new - f_est)
                f_est = f_new
                cnt += 1

                if cnt > 100:
                    raise RuntimeError('Did not converge \n Pipe:{0},\n V: {1:.2f} [m/s],\n f:{2:.4f} [-]'.format(
                        self._pipe, self._V, f_est))
        else:
            f_est = 1e-4

        return f_est

    @property
    def Re_d(self):
        """ Reynolds number

        :return:
        """
        return f_Re(self.points['in'].fluid.rho, self.points['in'].fluid.mu, self.V, self._pipe.d)

    def __eq__(self, other):
        return (self.points['in'] == other.points['in']) and (self.points['out'] == other.points['out'])

    def __str__(self):
        return '{4}->{5} V: {0:>6.3f} [m/s], Q: {1:>6.2f} [m3/h], f: {2:>6.4f} [-], Re_d: {3:>6.2e}'.format(self.V,
                                                                                                            self.Q * 3600,
                                                                                                            self.f,
                                                                                                            self.Re_d,
                                                                                                            self.points[
                                                                                                                'in'].name,
                                                                                                            self.points[
                                                                                                                'out'].name)


class PipeFlowList(object):

    def __init__(self):
        """ A list of PipeFlow Objects

        """
        self._list = []

    def get_flows_to_point(self, point:Point):
        """ Returns sublist which contains all flows to given point

        :param point:
        :return: PipeFlowList
        """
        flow_list = PipeFlowList()
        for fl in self:
            if point in fl.points.values():
                flow_list.append(fl)

        return flow_list

    def append(self, item: PipeFlow):
        """Add pipeflow"""
        if type(item) is PipeFlow:
            self._list.append(item)
        else:
            raise ValueError("Cannot add {0} to {1}".format(type(item), type(self)))

    def __index__(self, item: PipeFlow):
        ind = -1
        for i, pf in enumerate(self._list):
            if pf == item:
                ind = i
                break
        return ind

    def index(self, item: PipeFlow):
        """ Returns index of item

        :param item:
        :return:
        """
        return self.__index__(item)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index:int, value:PipeFlow):
        if type(value) is PipeFlow:
            self._list[index] = value
        else:
            raise ValueError("Cannot set {0} in {1}".format(type(value), type(self)))

    def __add__(self, other: PipeFlow):
        tmp = PipeFlowList()
        for pf in self:
            tmp.append(pf)

        if type(other) is PipeFlowList:
            for pf in other:
                tmp.append(other)
        else:
            raise ValueError("Cannot sum {0} and {1}".format(type(self), type(other)))

    def __iter__(self):
        return iter(self._list)


class JunctionConnection(object):

    def __init__(self, point, pipe):
        """ Defines a connection to a juction
        :param point:  Point which connects to junction
        :param pipe: Pipe which achieves the connection
        """
        self.point = point
        self.pipe = pipe

    def __str__(self):
        return 'to {0}, with {1}'.format(self.point, self.pipe)


class JunctionOptimizationInterface(object):
    """ An interface for junction optimization

    """

    def get_state_equation(self, isolate=False):
        raise NotImplementedError()

    def get_state_symbol(self):
        raise NotImplementedError()

    def get_state_value(self):
        raise NotImplementedError()

    def update_state(self, value):
        raise NotImplementedError()


class ProgressReport(object):

    def __init__(self, progress: int, message: str):

        self._progress = progress
        self._message = message

    @property
    def progress(self):
        return self._progress

    @property
    def message(self):
        return self._message


class JunctionSystemSolver(object):

    def __init__(self, update_rate=0.1, n_max=100, conv_thr=1e-4):
        """ Solver which can be used to compute flows in a system of pipes

        :param update_rate: The update rate of the solver
        :param n_max: Maximum number of iterations
        :param conv_thr: Convergence threshold for net-flow difference in between iterations

        """

        # Update rate
        self._gamma = update_rate

        # Convergence check:
        self._conv_thresh = conv_thr
        self._n_max = n_max

        # Symbolic variables:
        self._Asymb = None
        self._bsymb = None

        self._Aeq = None
        self._beq = None

        self._progress_report = mbb.EventPublisher(data_type=ProgressReport)

    @property
    def progress_event(self):
        return self._progress_report

    def solve(self, junctions, verbose=0, isolate_junctions=False, n_initializations=2):
        """ Solve flow for supplied (system) of junctions

        :param junctions: A list of junctions for which the pressures and flows need to be solved.
        :param verbose: Integer indicating the verbosity of the solver; 0 indicates silent, 1 basic, 2  detailed.
        :param isolate_junctions: Solve junctions in isolation (i.e. ignore inter-junction connections).
        :param n_initializations: Number of initialization steps (junctions computed in isolation)
        before starting to solve.
        :return:
        """
        # Initialize:
        # Used to initialize individual junctions close to their final value
        p_list = []
        self._progress_report.raise_event(ProgressReport(progress=0, message='Initializing'))
        if len(junctions) > 1:
            for n in range(n_initializations):
                self._progress_report.raise_event(ProgressReport(progress=int(100*n/n_initializations),
                                                                 message='Initializing {0}/{1}'.format(n+1, n_initializations)))
                if verbose > 0:
                    print('Initializing {0}/{1}...'.format(n + 1, n_initializations))
                for j in junctions:
                    j.initialize_states(verbose=int(verbose) - 1)
        else:
            isolate_junctions = True

        self._progress_report.raise_event(ProgressReport(progress=100,message='Finished initialization'))
        # Optimize:
        # Optimize coupled junctions to obtain zero net-flow at each junction
        n_it = 0
        Qsum = self._conv_thresh + 1
        while Qsum > self._conv_thresh and n_it < self._n_max:
            if np.mod(n_it, 1) == 0 and verbose > 0:
                print('Solving iteration {0}... (Qsum: {1:.2f})'.format(n_it + 1, Qsum if n_it > 0 else -1))

            self._progress_report.raise_event(ProgressReport(progress=0,
                                                             message='Solving It. {0}'.format(n_it + 1, self._n_max)))

            # Update symbolic expressions of A and B
            if np.mod(n_it, 1) == 0 and verbose > 1:
                print('- Updating system...')
            self._update_system(junctions, isolate_junctions=isolate_junctions)

            self._progress_report.raise_event(ProgressReport(progress=10,
                                                             message='It. {0}/{1}'.format(n_it + 1, self._n_max)))

            # Get linearization point
            if np.mod(n_it, 1) == 0 and verbose > 1:
                print('- Gathering linearization point...')
            h_tilde = self._get_htilde(junctions)

            self._progress_report.raise_event(ProgressReport(progress=15,
                                                             message='It. {0}/{1}'.format(n_it + 1, self._n_max)))

            # Get current flow value:
            if np.mod(n_it, 1) == 0 and verbose > 1:
                print('- Evaluating system...')
            A, b = self._evaluate_system(h_tilde)
            Qsum = np.abs(b).sum()
            self._progress_report.raise_event(ProgressReport(progress=50,
                                                             message='It. {0}/{1}'.format(n_it + 1, self._n_max)))

            if np.mod(n_it, 1) == 0 and verbose > 1:
                print('- Junction flow error {0:.2f} [m3/h]...'.format(Qsum * 3600))

            # Solve linear system of equations:
            if np.mod(n_it, 1) == 0 and verbose > 1:
                print('- Computing least-squares...')
            res = np.linalg.lstsq(A.astype(float), b.astype(float), rcond=None)
            dh = res[0]
            self._progress_report.raise_event(ProgressReport(progress=90,
                                                             message='It. {0}/{1}'.format(n_it + 1, self._n_max)))
            # Update pressures
            for i, junction in enumerate(junctions):
                junction.update_state(self._gamma * dh[i] + junction.head)

            # Collect pressure distribution
            p_list.append([j.pressure for j in junctions])

            self._progress_report.raise_event(ProgressReport(progress=100,
                                                             message='It. {0}/{1}'.format(n_it + 1, self._n_max)))

            n_it += 1


        if verbose > 0:
            if n_it < self._n_max:
                # Get linearization point
                h_tilde = self._get_htilde(junctions)

                # Get current flow value:
                A, b = self._evaluate_system(h_tilde)
                Qsum = np.abs(b).sum()

                print('Converged in {0} iterations'.format(n_it))
                print('Final net Junction flow {0:.2f} [m3/h]'.format(Qsum * 3600))
            else:
                print('Did not converge in {0} iterations'.format(n_it))

        self._progress_report.raise_event(ProgressReport(progress=100,
                                                         message='Done')
                                          )

        return p_list

    def _get_htilde(self, junctions):
        """ Get the state vector
        :param junctions:
        :return:
        """
        return [(junc.get_state_symbol(), junc.get_state_value()) for junc in junctions]

    def _evaluate_system(self, h_tilde):
        """ Evaluate the system of equations at h_tilde

        :param h_tilde:
        :return:
        """

        vals = {str(a): b for a, b in h_tilde}

        A = np.array(self._Aeq.eval_expr(vals))
        b = np.array(self._beq.eval_expr(vals))

        return A, b[:, 0]

    def _update_system(self, junctions, isolate_junctions):
        """ Update internal system equations based on list of junctions

        :param junctions:  List of junctions to solve
        :param isolate_junctions: If true inter-junction connections are ignored (which reduces convergence speed, but increases stability
        :return:
        """

        # Generate list of equations to solve
        Q_list = []

        # Sum of flow at junctions is 0
        for junction in junctions:
            Q_list.append(junction.get_state_equation(isolate_junctions))  # Add item to flow list for junction flow

        self._bsymb = -symp.Matrix(Q_list)
        self._beq = msym.EquationEvaluator(str(self._bsymb))

        # Collect symbolic variables :
        symbs = [junction.get_state_symbol() for junction in junctions]
        self._Asymb = symp.Matrix([[symp.diff(Q, sh) for sh in symbs] for Q in Q_list])  # Gradient expression
        self._Aeq = msym.EquationEvaluator(str(self._Asymb))


class ConstantFlowVelocity(Point):
    # Todo: Test/implement this functionality

    def __init__(self, V, pressure, z, fluid, name=''):
        Point.__init__(self, pressure, z, fluid, name=name)

        self._V = V

    @property
    def velocity(self):
        return self._V


class JunctionPoint(Point, JunctionOptimizationInterface):

    def __init__(self, pressure=0.0, z=0.0, fluid=None, name=''):
        """ A flow junction, i.e. a spot at which multiple flows merge.

        Due to the conservation of mass junctions are characterized by a zero net-flow: flow in == flow out

        :param pressure: Pressure at the junction
        :param z:
        :param fluid:
        :param name:
        """
        Point.__init__(self, pressure=pressure, z=z, fluid=fluid, name=name)
        self._connections = []

    def connect(self, point, pipe):
        """ Connect a point to this junction using the specified pipe

        :param point: Point to connect to
        :param pipe: Pipe which establishes connection
        :return:
        """

        # Verify input:
        if len(self._connections) == 0 and (self.fluid is None):  # Check if junction fluid is defined
            # No junction fluid defined, take point fluid
            self._fluid = point.fluid
        elif not (self.fluid == point.fluid):  # Verify fluid
            raise ValueError("Point fluid doesn't match Junction Fluid.")

        # Add connection:
        self._connections.append(JunctionConnection(point, pipe))

    def initialize_states(self, h_J0=0.0, **kwargs):
        """ Initialize junction values

        :param h_J0: Initial junction head
        :param kwargs:
        :return:
        """

        # Initialize using a model-based solver:
        solver = JunctionSystemSolver(update_rate=kwargs['update_rate'] if 'update_rate' in kwargs else 0.5)
        solver.solve([self], isolate_junctions=True,
                     verbose=kwargs['verbose'] if 'verbose' in kwargs else False)

        # Initialize using a model-free solver:
        # Find the roots of Qsum
        # res = optimize.root(self._Qsum, x0=h_J0)

    def get_state_symbol(self):
        """  Get symbolic expression for this junction head

        :return:
        """
        return self.symbolic_head

    def get_state_value(self):
        """ Get head value for this junction

        :return:
        """
        return self.head

    def update_state(self, value):
        """ Update junction pressure given head

        :return:  Head to match
        """
        self._match_pressure_to_head(value)
        return self.head

    def get_state_equation(self, isolate=False):
        """ Get flow equation for this junction

        :param isolate:
        :return:
        """

        Qsum = 0

        for connection in self.connections:
            # Express flow in symbolic junctions:
            symb_flow = self.get_symb_flow_with_connection(connection, isolate)  # Sum junction point

            # Add to flow list for junction
            Qsum += symb_flow

        return Qsum

    def get_symb_flow_with_connection(self, connection, isolate_junction=False):
        """ Get flow equation in symbolic expression

        :param connection:
        :param isolate_junction:
        :return:
        """

        if isinstance(connection.point, ConstantFlowVelocity):
            return connection.point.velocity * connection.pipe.A

        else:
            # Gather heads:
            if isinstance(connection.point, JunctionPoint) and not isolate_junction:
                h_point = connection.point.symbolic_head
            else:
                h_point = connection.point.head

            # Compute head:
            dh = (h_point - self.symbolic_head)

            # Check sign of current value:
            pf = PipeFlow(connection.pipe, self, connection.point)
            sign = np.sign(pf.head_loss)
            # dh += pf.minor_head_loss

            return sign * PipeFlow.compute_flow_velocity(connection.pipe, pf.f, dh * sign) * connection.pipe.A

        # Check current head sign:

    @property
    def Qsum(self):
        """ Net head for this junction

        :return:
        """
        return self._Qsum(self.head)

    def _Qsum(self, h_J):
        """
        Get cumalative joint flow [m3/s].
        :param h_J: Joint head
        """

        if type(h_J) is np.ndarray:
            h_J = h_J[0]

        # Update joint pressure to given joint head
        self._match_pressure_to_head(h_J)

        # Compute flow
        Qsum = 0
        Qsum_abs = 0
        for flow in self.pipe_flows:
            Q = flow.Q
            Qsum += Q
            Qsum_abs += abs(Q)
        if Qsum_abs < 1e-5:
            print('No flow')
        return Qsum

    def _match_pressure_to_head(self, h_J):
        """Matches the joint pressure to joint head h_J
        :param h_J: [m] Junction point head
        """
        self._pressure = (h_J - self.z) * (g * self.fluid.rho)

    @property
    def pipe_flows(self):
        """ List of pipeflows towards this junction

        :return:
        """
        flows = PipeFlowList()
        for conn in self._connections:
            flows.append(PipeFlow(conn.pipe, self, conn.point))
        return flows

    def __str__(self):
        conn_str = '----{0}---\n'.format(self.name)
        conn_str += 'p:{0:.2f} [kPa], Qsum: {1:.4f} [m3/h]\n'.format(self.pressure * 1e-3, self._Qsum(self.head))
        conn_str += '-------\n'
        for i, flow in enumerate(self.pipe_flows):
            conn_str += '{0}\n'.format(flow)
        conn_str += '-------'
        return conn_str

    @property
    def points(self):
        """List of points to which this junction connects

        :return:
        """
        return [c.point for c in self._connections]

    @property
    def connections(self):
        """List of junction connections

        :return:
        """
        return self._connections

    @property
    def n_connections(self):
        """The number of connections to this junction"""
        return len(self._connections)


class PunchHole(JunctionPoint, JunctionOptimizationInterface):

    def __init__(self, d_punch, l_punch, d_flow, l_flow, n_flow, p_ext, name='', close_punch_hole=False):
        """A wrapper class which imitates a typical punch hole

        :param d_punch: [m] Punch hole diameter
        :param l_punch: [m] Punch hole length
        :param d_flow: [m] diameter of flow hole
        :param l_flow: [m] length of flow hole
        :param n_flow: [-] number of flow holes
        :param p_ext: [Pa] pressure outside punch hole
        :param name: [-] Name of the punch hole (optional)
        :param close_punch_hole: [Bool] state of the punch hole (True: close punch hole, False: open punch hole)
        """
        JunctionPoint.__init__(self, pressure=p_ext.pressure * (1 - 0.001), z=p_ext.z, fluid=p_ext.fluid, name=name)

        self._n_flow = n_flow
        self._flowhole_pipe = Pipe(d=d_flow, L=l_flow, eps=0.01e-3, K=1.4)

        # if not close_punch_hole:
        self._punch_pipe = Pipe(d=d_punch, L=l_punch, eps=0.01e-3, K=1.4)
        # else:
        # self._punch_pipe = Pipe(d=d_punch*1e-2, L=l_punch, eps=0.01e-3, K=1.4)

        # Connect flow holes
        for n in range(n_flow):
            self.connect(p_ext, self._flowhole_pipe)

        # Connect punch hole
        if not close_punch_hole:
            self.connect(p_ext, self._punch_pipe)

    @property
    def n_flowholes(self):
        """ Number of flow holes

        :return:
        """
        return self._n_flow


class PunchSystem(object):

    def __init__(self, n_around: int, n_across: int, d_punch: float, l_punch: float, d_phole: float, l_phole: float,
                 d_bore: float, l_bore: float, l_shaft: float, n_flow_holes: int, d_flow_hole: float,
                 l_flow_hole: float, wall_roughness: float):

        self.n_around = n_around
        self.n_across = n_across
        self.d_punch = d_punch
        self.l_punch = l_punch
        self.d_phole = d_phole
        self.l_phole = l_phole
        self.d_bore = d_bore
        self.l_bore = l_bore
        self.l_shaft = l_shaft
        self.d_flow_hole = d_flow_hole
        self.l_flow_hole = l_flow_hole
        self.n_flow_holes = n_flow_holes
        self.wall_roughness = wall_roughness

        self._junctions = None
        self._penv = Point(pressure=1e5, z=0, fluid=air)
        self._ppump = Point(pressure=1e5, z=0, fluid=air)

        self._solver = JunctionSystemSolver(update_rate=0.8)
        self._update_junctions(False)

    @property
    def solver(self):
        return self._solver

    @property
    def junctions(self):
        return self._junctions

    def solve(self, p_pump, p_env=None, close_punch_holes=False, **kwargs):

        if p_env is None:
            p_env = Point(pressure=100e3, z=0, fluid=air, name='env')
        self._penv = p_env
        self._ppump = p_pump

        # Generate system
        self._update_junctions(close_punch_holes)

        # Create solver
        self._solver.solve(self._junctions, verbose=0)

    def _update_junctions(self, close_punch_holes):

        # Define pipes:
        pipe_shaft = Pipe(d=self.d_bore, L=self.l_shaft, eps=self.wall_roughness)
        pipe_bore = Pipe(d=self.d_bore, L=self.l_bore, eps=self.wall_roughness)
        pipe_phole = Pipe(d=self.d_phole, L=self.l_phole, eps=self.wall_roughness, K=1.4)

        # Create junctions:
        junctions = []
        for i in range(self.n_across):
            dv = np.random.randn() * 1e3
            junctions.append(JunctionPoint(pressure=self._penv.pressure + dv,  # Choose environmental pressure
                                           z=0.0, fluid=self._ppump.fluid, name='junc{0}'.format(i + 1)))

        # Make connections:
        for i in range(self.n_across):
            # Predecessor:
            if i == 0:
                # Connect shaft to first junction
                junctions[i].connect(point=self._ppump, pipe=pipe_shaft)
            else:
                junctions[i].connect(point=junctions[i - 1], pipe=pipe_bore)

            # Connection to next junction if exists:
            if i < (self.n_across - 1):
                junctions[i].connect(point=junctions[i + 1], pipe=pipe_bore)

            # Add connections to 'punch holes':
            for j in range(self.n_around):
                # Connect
                tmp_ph = PunchHole(d_punch=self.d_punch, l_punch=self.l_punch,
                                   d_flow=self.d_flow_hole, l_flow=self.l_flow_hole,
                                   n_flow=self.n_flow_holes,
                                   p_ext=self._penv,
                                   name='ph_{{{0},{1}}}'.format(i + 1, j + 1),
                                   close_punch_hole=close_punch_holes
                                   )
                # Connect
                tmp_ph.connect(point=junctions[i], pipe=pipe_phole)
                junctions[i].connect(point=tmp_ph, pipe=pipe_phole)

                #  Add to junctions
                junctions.append(tmp_ph)

        self._junctions = junctions

    def __str__(self):
        val_str = ''
        for key, item in self.__dict__.items():
            val_str += '{0:<15}:{1:>10}\n'.format(key, item)

        return val_str


if __name__ == "__main__":
    fc_6mm9x10 = PunchSystem(n_around=9,
                             n_across=10,
                             d_punch=6e-3,
                             l_punch=4e-3,
                             d_phole=18e-3,
                             l_phole=((372-110)/2 - 4)*1e-3,
                             d_bore=110e-3,
                             l_bore=162e-3,
                             l_shaft=447e-3,
                             d_flow_hole=3e-3,
                             n_flow_holes=4,
                             l_flow_hole=3e-3,
                             wall_roughness=0.01e-3
                             )

    def cb_progress(publisher):
        data = publisher.get_data()
        print(data.message)
    sub = mbb.SimpleEventSubscriber(h_callback=cb_progress)
    p_pump = Point(z=0, pressure=98e3,fluid=air)
    fc_6mm9x10.solver.progress_event.connect(sub)
    fc_6mm9x10.solve(p_pump)




