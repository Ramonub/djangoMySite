import numpy as np
import scipy as sp
import scipy.linalg
import traitlets, logging
from traitlets import CFloat, CInt, CUnicode
import os
from copy import copy, deepcopy
import xml.etree.cElementTree as ET
import madernpytools.backbone as mbb
import madernpytools.models.mechanics as mmech
from madernpytools.models.mechanics import CylindricHertzContact, IStiffnessElement, HertzContact
from madernpytools.models.mechanics import Rod, Thread, Tube, Material
import madernpytools.tools.symbolic_tools as msym
import madernpytools.module_data as mdata
import sympy as symp

_logger = logging.getLogger(f'madernpytools.{__name__}')


class MadernModelLibrary(object):

    def __init__(self):
        self._item_dict = {}

    def load(self, library_dir: str, append: bool = True, verbose=False):
        """ Load dictionary from files in library_dir.

        @param library_dir: Directory in which library files are stored
        @param append:  If true, items are added to existing library object, if false library is reset and only
        new values are added
        @param verbose: verbose Show comments
        @return:
        """

        if not append:
            self._item_dict = {}

        # list dir:
        flist = [file for file in os.listdir(library_dir) if os.path.splitext(file)[-1].lower() == '.xml']

        for f in flist:
            full_filename = '{0}/{1}'.format(library_dir, f)

            # Define key as filename:
            key = os.path.splitext(f)[0]

            if not (key in self._item_dict.keys()):
                try:
                    self._item_dict[key] = ModelItemLoader().load(full_filename)
                    if verbose:
                        print('Loaded : {0}'.format(f))
                except IOError:
                    # Todo; Is this pass correct?
                    pass
                except:
                    if verbose:
                        print('Failed loading {0}'.format(f))
            else:
                raise RuntimeWarning('Did not load {0}, item with same name is already in library'.format(f))

        return self

    def get_of_type(self, class_type):
        sub_dict = {}

        for key, item in self._item_dict.items():
            if isinstance(item, class_type):
                sub_dict[key] = item
        return sub_dict

    @property
    def items(self):
        return self._item_dict


def get_module_library():
    file_directory = '{0}/data/library'.format(mdata.pkg_path)
    return MadernModelLibrary().load(file_directory)


class ToolsetClassFactory(mbb.IClassFactory):

    @staticmethod
    def get(name):
        return eval(name)


class ModelItemLoader(object):

    def load(self, filename: str):
        root_elem = ET.parse(filename).getroot()
        root_type = ToolsetClassFactory().get(root_elem.get('Type'))
        return root_type.from_xml(root_elem, ToolsetClassFactory())


steel = Material(rho=7.8e3, E=200e3, v=0.27, material_name='Steel')
M10x15 = Thread(d=10, d_pitch=9.1, d_root=8.5, material=steel)

"""   TOOLSET STIFFNESS ELEMENTS
         
"""


class ToolsetBody(mmech.IKineticEnergy):

    def __init__(self, sub_script=''):
        subscript = id(self) if sub_script == '' else sub_script
        self.set_symbol_name(subscript)

    def set_symbol_name(self, name):
        self._symb = symp.symbols('x_{0}'.format(name))
        self._symb_d = symp.symbols('xd_{0}'.format(name))
        self._symb_dd = symp.symbols('xdd_{0}'.format(name))

    @property
    def mass(self):
        raise NotImplementedError()

    @property
    def symbolic_states(self) -> list:
        return [self._symb]

    @property
    def symbolic_state_derivatives(self) -> list:
        return [self._symb_d]

    @property
    def symbolic_state_2th_derivatives(self) -> list:
        return [self._symb_dd]

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return self.mass * self._symb_d ** 2

    def get_state_value(self, key) -> float:
        return 0.0


class DummyBody(ToolsetBody):

    def __init__(self, mass, subscript='dummy'):
        ToolsetBody.__init__(self, subscript)
        self._mass = mass

    @property
    def mass(self):
        return self._mass


class DummyStiffness(mmech.IStiffnessElement):

    k = CFloat()

    def __init__(self, k, *args, **kwargs):
        """

        @param k:
        """
        traitlets.HasTraits.__init__(self, k=k, *args, **kwargs)

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * self.k * state_eq ** 2


class SystemDynamics(object):

    def __init__(self, bodies: dict, body_connections: dict):

        self._bodies = bodies
        self._connections = body_connections

        self._T = 0  # Symbolic kinetic energy
        self._V = 0  # Symbolic potential energy

        # Update symbol names:
        for key, body in self._bodies.items():
            body.set_symbol_name(key)

        self._generate_eom()

    def collect_body_attribute(self, attr: str):
        q = []
        for _, item in self._bodies.items():
            q += getattr(item, attr)
        return q

    @property
    def q(self):
        """ Return body states

        @return:
        """
        return self.collect_body_attribute('symbolic_states')

    @property
    def qd(self):
        """ Return body state derivatives

        @return:
        """
        return self.collect_body_attribute('symbolic_state_derivatives')

    @property
    def qdd(self):
        """ Return body state 2th derivatives

        @return:
        """
        return self.collect_body_attribute('symbolic_state_2th_derivatives')

    @property
    def potential_energy(self):
        V = 0
        for _, item in self._bodies.items():
            # Collect energy equations:
            V += item.mass * item.symbolic_states[0]  # Potential

        for _, item in self._connections.items():
            V += item.get_energy_equation(None)

        return V

    @property
    def kinetic_energy(self):
        T = 0
        for key, item in self._bodies.items():
            # Collect energy equations:
            T += item.get_energy_equation(None)  # Kinetic

        return T

    def _generate_eom(self):

        T = self.kinetic_energy
        V = self.potential_energy
        q = self.q
        qd = self.qd

        # Mass matrix:
        Tqd = [symp.diff(T, qd_item) for qd_item in qd]  # dT/dqd
        self._Tqddt = symp.Matrix(
            [[symp.diff(col, qd_item) for qd_item in qd] for col in Tqd])  # d/dt(dT/dqd) -> M (mass matrix)

        # Stiffness matrix:
        Vq = symp.Matrix([symp.simplify(symp.diff(V, q_item)) for q_item in q])  # dV/dq
        self._Vqq = symp.simplify(
            symp.Matrix([[symp.diff(col, q_item) for q_item in q] for col in Vq]))  # d2V/dq2 -> K (stiffness matrix)

    @property
    def value_dict(self):
        val_dict = {}
        for _, item in self._bodies.items():
            for val in item.symbolic_states:
                val_dict[str(val)] = item.get_state_value(str(val))

            for val in item.symbolic_state_derivatives:
                val_dict[str(val)] = item.get_state_value(str(val))
        return val_dict

    @property
    def symbolic_mass_matrix(self) -> symp.Expr:
        return self._Tqddt

    @property
    def symbolic_stiffness_matrix(self) -> symp.Expr:
        return self._Vqq

    def mass_matrix(self, val_dict):
        return msym.EquationEvaluator(str(self.symbolic_mass_matrix)).eval_expr(val_dict)

    def stiffness_matrix(self, val_dict):
        return msym.EquationEvaluator(str(self.symbolic_stiffness_matrix)).eval_expr(val_dict)

    def mode_shapes(self, val_dict):
        K = np.array(self.stiffness_matrix(val_dict)).astype(float)
        M = np.array(self.mass_matrix(val_dict)).astype(float)
        return sp.linalg.eig(K, M)


class Ground(ToolsetBody):

    def __init__(self, sub_script=''):
        ToolsetBody.__init__(self, sub_script=sub_script)

    @property
    def mass(self):
        return 1e9

    @property
    def symbolic_states(self) -> list:
        return []

    @property
    def symbolic_state_derivatives(self) -> list:
        return []

    @property
    def symbolic_state_2th_derivatives(self) -> list:
        return []

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return symp.Expr(0.0)

    def get_state_value(self, key) -> float:
        pass


class SpacerElement(IStiffnessElement, mbb.MadernObject):
    info_text = "Spacer element"

    @property
    def k(self):
        raise NotImplementedError()

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * (self.k * 1e3) * state_eq ** 2


class TensionElement(IStiffnessElement, mbb.MadernObject):
    info_text = "Tension element"

    @property
    def k(self):
        """ Returns the stiffness in N/mm

        @return:
        """
        raise NotImplementedError()

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * (self.k * 1e3) * state_eq ** 2


class RiRoTensionScrew(TensionElement, mbb.TraitsXMLSerializer):
    info_text = "RiRo tension screw"
    illustration_path = f'{mdata.pkg_path}/data/figures/riro_tension_screw.png'
    l_loaded = CFloat(default_value=1.0)
    thread = M10x15

    def __init__(self, l_loaded: float, thread: Thread):
        """ RiRo toolsets are tensioned using a screw-connection

        @param l_loaded: The screw shaft length which is elongated
        @param thread: Thread size
        """
        mbb.TraitsXMLSerializer.__init__(self, l_loaded=l_loaded, thread=thread)

    @property
    def k(self):
        rod = Rod(self.l_loaded, d=self.thread.d_root, E=self.thread.material.E)
        return (self.thread.k ** -1 + rod.k ** -1) ** -1


class RiRoShimPack(SpacerElement, mbb.TraitsXMLSerializer):
    info_text = "RiRo shimpack"
    illustration_path = f'{mdata.pkg_path}/data/figures/riro_shimpack.png'

    surface_area = CFloat(1.0)
    thickness = CFloat(1.0)
    material = mmech.IMaterial()

    def __init__(self, surface_area: float, thickness: float, material: Material):
        """RiRo shim pack item

        @param surface_area: [mm2] Contact surface of shimpack
        @type surface_area: float
        @param thickness: [mm] Thickness of shim pack
        @type thickness: float
        @param material: Material of shim pack
        @type material: Material
        """
        mbb.TraitsXMLSerializer.__init__(self, surface_area=surface_area, thickness=thickness, material=material)

    @property
    def k(self):
        return self.material.E * self.surface_area / self.thickness


class SetScrew(SpacerElement, mbb.TraitsXMLSerializer):
    info_text = "Set screw"
    illustration_path = f'{mdata.pkg_path}/data/figures/bobst_setscrew.png'

    height = CFloat(help='Set screw height')
    diameter = CFloat(help='Set screw diameter')
    thread = traitlets.TraitType(help='Set screw thread')
    material = mmech.IMaterial(help='Set screw material')

    def __init__(self, height, diameter, thread: Thread, material: Material):
        """
        # TODO add description/drawing for this setscrew

        @param height:  Set screw height
        @param diameter:  Set screw (outer diameter)
        @param thread: Set screw thread
        @param material:  Set screw material
        """
        mbb.TraitsXMLSerializer.__init__(self, height=height, diameter=diameter, thread=thread, material=material)
        self._body = Rod(L=height, d=diameter, E=material.E)
        self.observe(handler=self._cb_observe)

    def _cb_observe(self, o):
        self._body = Rod(L=self.height, d=self.diameter, E=self.material.E)

    @property
    def k(self):
        return (self._body.k ** -1 + self.thread.k ** -1) ** -1


class SpacerNSetScrews(SpacerElement, mbb.TraitsXMLSerializer):
    illustration_path = f'{mdata.pkg_path}/data/figures/bobst_n_setscrews.png'
    info_text = "Parallel arangement of setscrews"
    set_screw = SetScrew(height=100, diameter=30, thread=M10x15, material=steel)
    n = CInt()

    def __init__(self, set_screw: SetScrew, n=2):
        mbb.TraitsXMLSerializer.__init__(self, set_screw=set_screw, n=n)

    @property
    def k(self):
        return self.set_screw.k * self.n


class TensionRod(TensionElement, mbb.TraitsXMLSerializer):
    illustration_path = f'{mdata.pkg_path}/data/figures/tension-rod.png'
    l_rod = CFloat()
    d_rod = CFloat()
    l_tube = CFloat()
    d_itube = CFloat()
    d_otube = CFloat()
    thread = traitlets.TraitType()
    material = steel

    def __init__(self, l_rod: float, d_rod: float, l_tube: float, d_itube: float, d_otube: float,
                 thread: Thread, material: Material):
        """

        @param l_rod:  Length of the tie-rod
        @param d_rod:
        @param d_itube:
        @param d_otube:
        @param thread:
        @param material:
        """

        # todo: Currently we implement a single length, while the length of tube and rod are different in practice
        mbb.TraitsXMLSerializer.__init__(self, l_rod=l_rod, d_rod=d_rod, l_tube=l_tube, d_itube=d_itube, d_otube=d_otube,
                                         thread=thread,
                                         material=material)

        # Derived elements:
        # Tube:
        self._tube = Tube(d_inner=d_itube, d_outer=d_otube, L=l_tube, E=material.E)

        # Rod:
        self._rod = Rod(L=l_rod, d=d_rod, E=material.E)

        self.observe(self._cb_handler)

    @property
    def k(self):
        return (self._tube.k ** -1 + self._rod.k ** -1 + 2 * self.thread.k ** -1) ** -1

    def _cb_handler(self, o):
        # Tube:
        self._tube = Tube(d_inner=self.d_itube, d_outer=self.d_otube, L=self.l_tube, E=self.material.E)

        # Rod:
        self._rod = Rod(L=self.l_rod, d=self.d_rod, E=self.material.E)


class TensionNTieRods(TensionElement, mbb.TraitsXMLSerializer):
    illustration_path = ''
    tie_rod = TensionElement()
    n = CFloat()

    def __init__(self, tie_rod, n=2):
        mbb.TraitsXMLSerializer.__init__(self, tie_rod=tie_rod, n=n)

    @property
    def k(self):
        return self.tie_rod.k * self.n


"""   BEARINGS 

"""

# TODO: Bearing load and name are generic properties, and should be moved from implementers of IBearing to IBearing
# TODO: this involves updating the existing library

class IBearing(mbb.TraitsXMLSerializer, IStiffnessElement, mbb.MadernObject):
    illustration_path = ''

    @property
    def k(self):
        """Radial stiffness [N/mm]"""
        raise NotImplementedError()

    @property
    def k_theta(self):
        """
        Rotational stiffness [Nmm/rad]
        """
        raise NotImplementedError()

    def stiffness(self, load):
        """Radial stiffness [Nmm/rad]"""
        raise NotImplementedError()

    @property
    def load(self):
        """Radial Load [N]"""
        raise NotImplementedError()

    @load.setter
    def load(self, value):
        raise NotImplementedError()

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        raise NotImplementedError()


class TaperedBearing(IBearing):
    info_text = "Tapered bearing"
    material = mmech.IMaterial
    illustration_path = ''
    r_inner = CFloat()
    r_outer = CFloat()
    r_roller = CFloat()
    contact_length = CFloat()
    n_rollers = CInt()
    body_angle = CFloat()
    load = CFloat()
    name = CUnicode()

    def __init__(self, material: Material, r_inner, r_outer, r_roller, contact_length, n_rollers, body_angle, load=0.0,
                 name=''):
        """
        @param material: Bearing material
        @param r_inner: [mm] radius of inner ring at roller body contact
        @param r_outer: [mm] radius of outer ring at roller body contact
        @param r_roller : [mm] radius of rotating body
        @param contact_length: contact length of rolling bodies
        @param n_rollers: number of rolling bodies
        @param body_angle: body angle [rad]
        @param load: Bearing load [kN]
        @param name: Name assigned to this bearing
        """
        mbb.TraitsXMLSerializer.__init__(self, material=material, r_inner=r_inner, r_outer=r_outer, r_roller=r_roller,
                                         contact_length=contact_length,
                                         n_rollers=n_rollers, body_angle=body_angle, load=load, name=name)

        self.observe(handler=self.cb_load_change, names='load')

    @traitlets.observe('load')
    def cb_load_change(self, change):
        pass
        #print('Changed from {0} -> {1}'.format(change['old'], change['new']))

    @property
    def k(self):
        """Bearing stiffness at current load

        @return:
        """
        return self.stiffness(self.load)

    @property
    def k_theta(self):
        """ Rotational stiffness [Nmm/rad]

        """
        return 0.0

    def stiffness(self, F):
        """ Bearing stiffness at given load

        @param F: Force applied to bearing
        @return:
        """

        F = abs(F)  # Bearings always have stiffness (except at zero)

        # We assume half the bodies are in contact:
        n_contacts = self.n_rollers / 2
        beta = self.body_angle

        angles = np.linspace(-np.pi / 2, np.pi / 2, int(n_contacts))
        alpha = np.cos(angles) ** 2 / (np.cos(angles) ** 2).sum()

        # Contact bodies are assumed to be connected in parallel, therefore the bearing stiffness is their weighted sum
        k_total = 0
        for a in alpha:
            k_total += self._rolling_body_stiffness(F * a * np.cos(beta))

        return k_total

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * (self.k * 1e3) * state_eq ** 2

    def _rolling_body_stiffness(self, F):
        """ Compute the stiffness of a single bearing body in between the outer and inner rings N/mm

        @param F: Applied force
        @return:
        """

        E = self.material.E
        v = self.material.v
        r_inner = self.r_inner
        r_outer = self.r_outer
        r_roller = self.r_roller
        l_contact = self.contact_length

        inner_contact = CylindricHertzContact(R_1=r_inner, R_2=r_roller, L=l_contact, E_1=E, E_2=E, v_1=v, v_2=v)
        outer_contact = CylindricHertzContact(R_1=-r_outer, R_2=r_roller, L=l_contact, E_1=E, E_2=E, v_1=v, v_2=v)

        k_outer = outer_contact.stiffness(F)
        k_inner = inner_contact.stiffness(F)

        return (k_inner ** -1 + k_outer ** -1) ** -1


class OTaperedBearing(IBearing):
    bearing = IBearing()
    effective_distance = CFloat()
    load = CFloat()
    name = CUnicode()

    def __init__(self, bearing: TaperedBearing, effective_distance, load=0.0, name=''):
        mbb.TraitsXMLSerializer.__init__(self, bearing=bearing, effective_distance=effective_distance,
                                         load=load, name=name)

    @traitlets.observe('load')
    def _cb_load_change(self, o):
        self.bearing.load = o['new'] / 2

    @property
    def k(self):
        return 2 * self.bearing.k

    @property
    def k_theta(self):
        """ Rotational stiffness [Nmm/rad]

        """
        return self.k * self.effective_distance ** 2 / 2

    def stiffness(self, F):
        return 2 * self.bearing.stiffness(F / 2)

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * (self.k * 1e3) * state_eq ** 2


class SphericalBearing(IBearing):
    material = mmech.IMaterial()
    r_outer = CFloat()
    r_inner = CFloat()
    r_roller = CFloat()
    r_roller2 = CFloat()
    n_rollers = CInt()
    load = CFloat()
    k_theta = CFloat()
    name = CUnicode()

    def __init__(self, material: Material, r_outer, r_inner, r_roller, r_roller2, n, load=0.0, k_theta=0.0,
                 name=''):
        """
        @param material: Bearing material
        @param r_inner: [mm] radius of inner ring
        @param r_outer: [mm] radius of outer ring
        @param r_roller : [mm] radius of rotating body
        @param r_roller2: [mm] second radius of rotating body
        @param n: number of rolling bodies
        @param load: bearing load [N]
        @param k_theta: bending stiffness [Nmm/rad]
        """

        mbb.TraitsXMLSerializer.__init__(self, material=material, r_outer=r_outer, r_inner=r_inner, r_roller=r_roller,
                                         r_roller2=r_roller2, n_rollers=n, load=load, k_theta=k_theta, name=name)

        self._body_angle = 0 / 180 * np.pi

    @property
    def k(self):
        """Bearing stiffness at current load

        @return:
        """
        return self.stiffness(self.load)

    def stiffness(self, F):
        """ Bearing stiffness at given load

        @param F: Force applied to bearing
        @return:
        """

        # We assume half the bodies are in contact:
        n_contacts = self.n_rollers / 2
        beta = self._body_angle

        angles = np.linspace(-np.pi / 2, np.pi / 2, int(n_contacts))
        alpha = np.cos(angles) ** 4 / (np.cos(angles) ** 4).sum()

        # Contact bodies are assumed to be connected in parallel, therefore the bearing stiffness is their weighted sum
        k_total = 0
        for a in alpha:
            k_total += self._rolling_body_stiffness(F * a * np.cos(beta))

        return k_total

    def _rolling_body_stiffness(self, F):
        """ Compute the stiffness of a single bearing body inbetween the outer and inner rings

        @param F: Applied force
        @return:
        """

        outer_contact = HertzContact(R_x1=-self.r_outer, R_x2=-self.r_roller2,  # Outer race
                                     R_y1=self.r_roller, R_y2=self.r_roller2 * 0.5,  # Rolling body
                                     E_1=self.material.E,
                                     E_2=self.material.E,
                                     v_1=self.material.v,
                                     v_2=self.material.v
                                     )
        inner_contact = HertzContact(R_x1=self.r_inner, R_x2=-self.r_roller2,  # Outer race
                                     R_y1=self.r_roller, R_y2=self.r_roller2 * 0.5,  # Rolling body
                                     E_1=self.material.E,
                                     E_2=self.material.E,
                                     v_1=self.material.v,
                                     v_2=self.material.v
                                     )

        # Divide by load (two rows of bearers)
        k_outer = outer_contact.stiffness(F / 2)
        k_inner = inner_contact.stiffness(F / 2)

        return 2 * ((k_inner ** -1 + k_outer ** -1) ** -1)

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return 0.5 * (self.k * 1e3) * state_eq ** 2


class Shaft(mbb.TraitsXMLSerializer, mbb.MadernObject):
    length = CFloat()
    outer_diameter = CFloat()
    inner_diameter = CFloat()
    material = mmech.IMaterial()
    illustration_path=''


    def __init__(self,
                 length: float,
                 outer_diameter: float,
                 inner_diameter: float,
                 material: Material
                 ):
        """ A circular shaft object

        @param length:  Shaft length, mm
        @param outer_diameter: Outer diameter of the shaft, mm
        @param inner_diameter: Inner diameter of the shaft (0 <= inner_diameter < outer_diameter, mm
        @param material: Shaft material
        """

        mbb.TraitsXMLSerializer.__init__(self, length=length, outer_diameter=outer_diameter,
                                         inner_diameter=inner_diameter,
                                         material=material)

    @traitlets.validate('outer_diameter', 'inner_diameter')
    def _validate_item(self, proposal):
        if proposal['value'] >= 0.0:
            return proposal['value']
        raise traitlets.TraitError("{1} should be larger than {0}".format(0.0, proposal.trait.name))

    @property
    def Ixx(self) -> float:
        """ Moment of inertia kg * mm^2

        @return:
        """
        return self.mass * (
                3 * ((self.outer_diameter / 2) ** 2 + (self.inner_diameter / 2) ** 2) + self.length ** 2) / 12

    @property
    def Iyy(self) -> float:
        """ Moment of inertia kg * mm^2

        @return:
        """
        return self.Ixx

    @property
    def Izz(self) -> float:
        """ Moment of inertia kg * mm^2

        @return:
        """
        return self.mass * ((self.outer_diameter / 2) ** 2 + (self.inner_diameter / 2) ** 2) / 2

    @property
    def second_area_moment(self) -> float:
        """ Second moment of area, also known as Area moment of inertia (mm^4)

        @return:
        """
        return np.pi * (self.outer_diameter ** 4 - self.inner_diameter ** 4) / 64

    @property
    def mass(self) -> float:
        """ Shaft mass (kg)

        @return:
        """
        return self.material.rho * (0.25 * np.pi * self.length * (self.outer_diameter ** 2
                                                                  - self.inner_diameter ** 2)) * 1e-9


class Gear(mbb.TraitsXMLSerializer, mbb.MadernObject, ToolsetBody):
    illustration_path = f'{mdata.pkg_path}/data/figures/gear.png'
    width = CFloat()
    d_pitch = CFloat()
    d_bore = CFloat()
    material = mmech.IMaterial
    module = CInt()

    def __init__(self, width: float, d_pitch: float, d_bore: float, material: Material, module: float):
        # Todo: Creating the shaft object before XML serialization might cause issues with serialization, verify this!

        mbb.TraitsXMLSerializer.__init__(self, width=width, d_pitch=d_pitch, d_bore=d_bore,
                                         material=material, module=module)
        ToolsetBody.__init__(self, sub_script='g')

        # Define shaft, and observe internal properties to ensure its values keep updated:
        self._shaft = Shaft(length=width, outer_diameter=d_pitch, inner_diameter=d_bore, material=material)
        self.observe(handler=self._cb_attr_change)

    def _cb_attr_change(self, val):
        self._shaft = Shaft(length=self.width, outer_diameter=self.d_pitch,
                            inner_diameter=self.d_bore, material=self.material)

    @property
    def mass(self):
        return self._shaft.mass


class BearerRing(mbb.TraitsXMLSerializer, mbb.MadernObject):
    info_text = "Bearer ring"

    width = CFloat()
    diameter = CFloat()
    angle = CFloat()
    material = mmech.IMaterial()
    thickness = CFloat()

    def __init__(self,
                 width: float,
                 diameter: float,
                 angle: float,
                 material: Material,
                 thickness: float = 0.0
                 ):
        """ Bearer ring object

        @param width:  Width of the bearer ring (mm)
        @param diameter:Outer (nominal) bearer diameter (mm)
        @param angle: conical angle of the bearer (deg)
        @param material: Bearer ring material
        @param thickness: Thickness of the bearer ring (currently not used, mm)
        """

        mbb.TraitsXMLSerializer.__init__(self, width=width, diameter=diameter, angle=angle,
                                         material=material, thickness=thickness)

    @property
    def nominal_bore_diameter(self):
        """ Returns the nominal bore diameter (i.e. diameter - 2 * thickness)

        @return:
        """
        return self.diameter - 2 * self.thickness


class BearerRingContact(mbb.TraitsXMLSerializer, IStiffnessElement, traitlets.TraitType):
    _ring_A = traitlets.TraitType()
    _ring_B = traitlets.TraitType()
    load = CFloat(0.0)

    def __init__(self, bearer_ring_A: BearerRing, bearer_ring_B: BearerRing):
        """ Simulation of the contact between two bearerrings A and B

        @param bearer_ring_A:  Bearer rings A
        @param bearer_ring_B: Bearer rings A
        """

        mbb.TraitsXMLSerializer.__init__(self, _ring_A=bearer_ring_A, _ring_B=bearer_ring_B)

        L = self._ring_A.width if self._ring_A.width < self._ring_B.width else self._ring_B.width

        self._hertz_contact = CylindricHertzContact(R_1=self._ring_A.diameter / 2,
                                                    R_2=self._ring_B.diameter / 2,
                                                    L=L,
                                                    E_1=self._ring_A.material.E,
                                                    E_2=self._ring_B.material.E,
                                                    v_1=self._ring_A.material.v,
                                                    v_2=self._ring_B.material.v)

        self._ring_A.observe(handler=self._cb_ring_change, names=['width', 'diameter', 'angle', 'material'])
        self._ring_B.observe(handler=self._cb_ring_change, names=['width', 'diameter', 'angle', 'material'])

    @traitlets.observe('_ring_A', '_ring_B')
    def _cb_ring_change(self, o):
        #print('Reacted to change')
        L = self._ring_A.width if self._ring_A.width < self._ring_B.width else self._ring_B.width
        self._hertz_contact = CylindricHertzContact(R_1=self._ring_A.diameter / 2,
                                                    R_2=self._ring_B.diameter / 2,
                                                    L=L,
                                                    E_1=self._ring_A.material.E,
                                                    E_2=self._ring_B.material.E,
                                                    v_1=self._ring_A.material.v,
                                                    v_2=self._ring_B.material.v)

    def indentation(self, F: float):
        """ Bearer ring contact indentation under load F (mm)

        @param F: Load on bearer contact (N)
        @return:
        """
        return self._hertz_contact.indentation(F)

    def stiffness(self, F: float):
        """ Bearer ring contact stiffness under load F  (N/mm)

        @param F: Load on bearer contact (N)
        @return:
        """
        return self._hertz_contact.stiffness(F)

    @property
    def k(self):
        return self.stiffness(self.load)

    def get_energy_equation(self, state_eq: symp.Expr):
        return 0.5 * (self.k * 1e3) * state_eq ** 2


"""   TOOLSET Model Interfaces 

"""


class IProductLayout(mbb.TraitsXMLSerializer, mbb.MadernObject):

    @property
    def max_axial_line(self):
        """ Maximum axial line length which simultaneously cuts
        @return:
        """
        raise NotImplementedError()


class IBearingBlock(mbb.TraitsXMLSerializer, mbb.MadernObject):

    bearing = IBearing()
    mass = traitlets.CFloat()

    @property
    def stiffness(self) -> float:
        raise NotImplementedError()


class ICylinder(mbb.TraitsXMLSerializer, mbb.MadernObject, mmech.IKineticEnergy):
    bearer_ring = traitlets.TraitType()
    os_shaft = traitlets.TraitType()
    body = traitlets.TraitType()
    ds_shaft = traitlets.TraitType()
    br_location = CFloat()

    @property
    def mass(self) -> float:
        raise NotImplementedError()

    @property
    def Ixx(self) -> float:
        raise NotImplementedError()

    @property
    def Iyy(self) -> float:
        raise NotImplementedError()

    @property
    def Izz(self) -> float:
        raise NotImplementedError()


class ICylinderLoads(mbb.MadernObject):

    def get_Fbr(self) -> float:
        """ Get cylinder bearer ring load (N)

        @return:
        """
        raise NotImplementedError()

    def get_Fbearing(self) -> float:
        """ Get cylinder bearing load (N)

        @return:
        """
        raise NotImplementedError()

    def get_qcut(self) -> float:
        """ Get cylinder cutting load  (N/mm)

        @return:
        """
        raise NotImplementedError()

    def get_qm(self) -> float:
        """ Get cylinder distributed load due to cylinder mass (N/mm)

        Note this value excludes the mass of the shafts

        @return:
        """
        raise NotImplementedError()

    def get_Mshaft(self) -> float:
        """ Get cylinder shaft moment (N*mm)

        @return:
        """
        raise NotImplementedError()

    def __neg__(self):
        """ Change position of cylinder from upper to lower

        @return:
        """
        raise NotImplementedError()


# Recursive relation between deflection and load
# For this reason we declare related Interfaces before defining them:
class IToolsetDeflections(object):
    pass


class IToolsetLoads(traitlets.HasTraits):
    F_t = CFloat()
    upper_cylinder_loads = ICylinderLoads()
    lower_cylinder_loads = ICylinderLoads()


class ICylinderDeflections(object):

    def update(self, loads: ICylinderLoads):
        raise NotImplementedError()

    def get_deflection(self, x: float):
        raise NotImplementedError()

    def get_rotation(self, x: float):
        raise NotImplementedError()


class IToolsetDeflections(object):

    def update(self, loads: IToolsetLoads):
        raise NotImplementedError()

    @property
    def upper_cylinder(self) -> ICylinderDeflections:
        raise NotImplementedError()

    @property
    def lower_cylinder(self) -> ICylinderDeflections:
        raise NotImplementedError()

    @property
    def bearer_deflection(self):
        raise NotImplementedError()

    def gap_deflection(self, x):
        raise NotImplementedError()


class IToolset(mbb.TraitsXMLSerializer, mbb.MadernObject):
    upper_cylinder = ICylinder()
    lower_cylinder = ICylinder()

    upper_bearing_block = IBearingBlock()
    lower_bearing_block = IBearingBlock()

    upper_gear = traitlets.TraitType()
    lower_gear = traitlets.TraitType()

    layout = IProductLayout()
    spacer = SpacerElement()
    tensioner = TensionElement()

    name = CUnicode()

    @property
    def loads(self) -> IToolsetLoads:
        raise NotImplementedError()

    @property
    def deflections(self) -> IToolsetDeflections:
        raise NotImplementedError()

    @property
    def gap_stiffness(self) -> float:
        raise NotImplementedError()

    def generate_dynamics(self):
        raise NotImplementedError()


""" Toolset Definition

"""


class Cylinder(ICylinder, ToolsetBody):
    illustration_path = f'{mdata.pkg_path}/data/figures/cylinder_dimensions.png'
    os_shaft = Shaft(length=1.0, outer_diameter=1.0, inner_diameter=0.5, material=steel)
    body = Shaft(length=1.0, outer_diameter=1.0, inner_diameter=0.5, material=steel)
    ds_shaft = Shaft(length=1.0, outer_diameter=1.0, inner_diameter=0.5, material=steel)
    bearer_ring = BearerRing(width=1.0, diameter=1.0, angle=1.0, material=steel, thickness=0.5)
    br_location = CFloat()

    def __init__(self,
                 os_shaft: Shaft,
                 ds_shaft: Shaft,
                 body: Shaft,
                 bearer_ring: BearerRing,
                 br_location: float
                 ):
        mbb.TraitsXMLSerializer.__init__(self, os_shaft=os_shaft, ds_shaft=ds_shaft, body=body, bearer_ring=bearer_ring,
                                         br_location=br_location)
        ToolsetBody.__init__(self)
        self.set_symbol_name(id(self))
        # Add cylinder rotation around web-directional axis:

    def set_symbol_name(self, name):
        self._symb = symp.symbols('x_{0}'.format(name))
        self._symb_d = symp.symbols('xd_{0}'.format(name))
        self._symb_dd = symp.symbols('xdd_{0}'.format(name))

        self._theta = symp.symbols('th_{0}'.format(name))
        self._theta_d = symp.symbols('thd_{0}'.format(name))
        self._theta_dd = symp.symbols('thdd_{0}'.format(name))

    @property
    def symbolic_states(self) -> list:
        return [self._symb, self._theta]

    @property
    def symbolic_state_derivatives(self) -> list:
        return [self._symb_d, self._theta_d]

    @property
    def symbolic_state_2th_derivatives(self) -> list:
        return [self._symb_dd, self._theta_dd]

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return self.mass * self._symb_d ** 2 + self.Ixx * self._theta_d ** 2

    def get_state_value(self, key) -> float:
        return 0.0

    @property
    def mass(self) -> float:
        # Note: As the body does have a cavity to mount the bearerrings, we do not include the bearer ring weight.
        # In other words: The bearerring weight is already included in the body into the body weight
        return self.os_shaft.mass + self.ds_shaft.mass + self.body.mass

    @property
    def Ixx(self):
        return (self.body.Ixx
                + (self.ds_shaft.Ixx + self.ds_shaft.mass * (self.ds_shaft.length + self.body.length / 2) ** 2)
                + (self.os_shaft.Ixx + self.os_shaft.mass * (self.os_shaft.length + self.body.length / 2) ** 2)
                )

    @property
    def Iyy(self):
        return self.Ixx

    @property
    def Izz(self):
        return self.body.Izz + self.os_shaft.Izz, self.ds_shaft.Izz


class BearingBlock(IBearingBlock, ToolsetBody):
    illustration_path = f'{mdata.pkg_path}/data/figures/bb_dimensions.png'
    bearing = IBearing()
    s = CFloat()
    mass = CFloat()

    def __init__(self, bearing: IBearing, s, mass):
        """

        @param bearing: Bearing mounted in the bearing block
        @param s: Distance between the body edge and the load location of the bearing (arrangement), mm
        """

        mbb.TraitsXMLSerializer.__init__(self, bearing=bearing, s=s, mass=mass)
        ToolsetBody.__init__(self, sub_script='bb')

    @property
    def stiffness(self):
        raise NotImplementedError()


class SimpleCylinderLoads(ICylinderLoads, traitlets.HasTraits):
    F_br = CFloat()
    F_bearing = CFloat()
    q_cut = CFloat()
    q_m = CFloat()
    M_shaft = CFloat()

    def __init__(self, F_br: float, F_bearing: float, q_cut: float, q_m: float, M_shaft):
        """
                                  __q_c_+_q_m_
                            _q_m_|     |    |
                           |     |     |   |
                           v_    v     v   |
                          |  |---^----^----|
                   __|o|__|  |             |
     Mext  /      |__   __|  |             |
           \        |o|   |  |             |
             ->      ^    |__|---v----v----|
                     |     |
                    F_s    v
                          F_br

        @param F_br:
        @param F_bearing:
        @param q_cut:
        @param q_m:
        @param M_shaft:
        """

        traitlets.HasTraits.__init__(self, F_br=F_br, F_bearing=F_bearing, q_cut=q_cut, q_m=q_m, M_shaft=M_shaft)

    def get_Fbr(self) -> float:
        """ Get cylinder bearer ring load (N)

        @return:
        """
        return self.F_br

    def get_Fbearing(self) -> float:
        """ Get cylinder bearing load (N)

        @return:
        """
        return self.F_bearing

    def get_qcut(self) -> float:
        """ Get cylinder cutting load  (N/mm)

        @return:
        """
        return self.q_cut

    def get_qm(self) -> float:
        """ Get cylinder distributed load due to cylinder mass (N/mm)

        Note this value excludes the mass of the shafts

        @return:
        """
        return self.q_m

    def get_Mshaft(self) -> float:
        """ Get cylinder shaft moment (N*mm)

        @return:
        """
        return self.M_shaft

    def __neg__(self):
        """
        Change load directions from upper to lower (male to female)

        i.e. negation causes a swap between:
                            ________q_m_____
                           |     |     |
                           v_    v     v   |
                          |  |---^----^----|
             ->    __|o|__|  |             |
     Mext  /      |__   __|  |             |
           \        |o|   |  |             |
                     |    |__|---v----v----|
                     v     ^     ^   ^   ^  |
                    F_s    |     |___|___|__|
                          F_br     q_cut

        and
                                  __q_c_+_q_m_
                            _q_m_|     |    |
                           |     |     |   |
                           v_    v     v   |
                          |  |---^----^----|
                   __|o|__|  |             |
     Mext  /      |__   __|  |             |
           \        |o|   |  |             |
             ->      ^    |__|---v----v----|
                     |     |
                    F_s    v
                          F_br


        @return:
        """
        return SimpleCylinderLoads(F_br=-self.F_br,  # Bearer ring force will come from the other direction
                                   F_bearing=-self.F_bearing,  # Bearing force will come from the other direction
                                   q_cut=-self.q_cut,  # Cutting force will come from the other direction
                                   q_m=self.q_m,  # Gravity will not flip (gravity always points downwards for both)
                                   M_shaft=-self.M_shaft)  # Torque will swap direction

    def __str__(self):
        string_format = "qm      : {0:>5.2f},\n"
        string_format += "q_cut   : {1:>5.2f},\n"
        string_format += "M_shaft :{2:>5.2f},\n"
        string_format += "F_br    : {3:>5.2f}\n"
        string_format += "F_bearing: {4:>5.2f}"
        return string_format.format(self.get_qm(), self.get_qcut(),
                                    self.get_Mshaft(), self.get_Fbr(), self.get_Fbearing())


class SimpleLayout(IProductLayout):
    ax_line_length = CFloat()

    def __init__(self, ax_line_length: float):
        mbb.TraitsXMLSerializer.__init__(self, ax_line_length=ax_line_length)

    @property
    def max_axial_line(self) -> float:
        return self.ax_line_length


class CylinderDeflectionModel(ICylinderDeflections):

    def __init__(self, cylinder: ICylinder, bearing_block: IBearingBlock, loads: ICylinderLoads, layout: IProductLayout,
                 *args, **kwargs):
        """
        @param cylinder : Cylinder of which we compute the bending
        @param loads : Loads applied on the cylinder
        @param bearing_block: bearing block mounted to the cylinder
        @param dimension_model: Deflection model dimensions object

                          __q_c + q_m__  F_br
                             |    |   |   |
                         |   v    v   v  _v
                         |----^----^----|  |
              ->      ^  |              |  |__|o|__
            /     V_a |  |              |  |__   __|
        M_a \_        |  |              |  |  |o|
                         |----v----v----|__|
                         |                     ^
                                               |
                                              F_s
        """

        self._cylinder = cylinder  # mbb.ArgumentVerifier(Cylinder, None).verify(cylinder)
        self._loads = loads  # mbb.ArgumentVerifier(AbstractCylinderLoads, None) .verify(loads)
        self._bb = bearing_block  # mbb.ArgumentVerifier(BearingBlock, None).verify(bearing_block)
        self._layout = layout

        self._val_dict = {}

        # Equations in string format:
        # TODO Move equation to external equation file:
        self._eq_dict = {
            'phi_xI': '(-M_a*x - V_a*x**2/2 + x**3*(q_c/6 + q_m/6))/(E_1*I_1)',
            'phi_xII': '(a**3*q_c/6 + q_m*x**3/6 + x**2*(-V_a/2 + a*q_c/2) + x*(-M_a - a**2*q_c/2))/(E_1*I_1)',
            'phi_xIII': '(F_br*b**2/2 + a**3*q_c/6 + q_m*x**3/6 + x**2*(F_br/2 - V_a/2 + a*q_c/2) + x*(-F_br*b - M_a - a**2*q_c/2))/(E_1*I_1)',
            'phi_xIIII': '(x**2*(F_br/2 - V_a/2 + a*q_c/2 + c*q_m/2) + x*(-F_br*b - M_a - a**2*q_c/2 - c**2*q_m/2) + (F_br*I_1*b*c - F_br*I_1*c**2/2 + F_br*I_2*b**2/2 - F_br*I_2*b*c + F_br*I_2*c**2/2 + I_1*M_a*c + I_1*V_a*c**2/2 + I_1*a**2*c*q_c/2 - I_1*a*c**2*q_c/2 - I_2*M_a*c - I_2*V_a*c**2/2 + I_2*a**3*q_c/6 - I_2*a**2*c*q_c/2 + I_2*a*c**2*q_c/2 + I_2*c**3*q_m/6)/I_1)/(E_1*I_2)',
            'v_xI': '(-M_a*x**2/2 - V_a*x**3/6 + x**4*(q_c/24 + q_m/24))/(E_1*I_1)',
            'v_xII': '(-a**4*q_c/24 + a**3*q_c*x/6 + q_m*x**4/24 + x**3*(-V_a/6 + a*q_c/6) + x**2*(-M_a/2 - a**2*q_c/4))/(E_1*I_1)',
            'v_xIII': '(-F_br*b**3/6 - a**4*q_c/24 + q_m*x**4/24 + x**3*(F_br/6 - V_a/6 + a*q_c/6) + x**2*(-F_br*b/2 - M_a/2 - a**2*q_c/4) + x*(F_br*b**2/2 + a**3*q_c/6))/(E_1*I_1)',
            'v_xIIII': '(x**3*(F_br/6 - V_a/6 + a*q_c/6 + c*q_m/6) + x**2*(-F_br*b/2 - M_a/2 - a**2*q_c/4 - c**2*q_m/4) + x*(F_br*I_1*b*c - F_br*I_1*c**2/2 + F_br*I_2*b**2/2 - F_br*I_2*b*c + F_br*I_2*c**2/2 + I_1*M_a*c + I_1*V_a*c**2/2 + I_1*a**2*c*q_c/2 - I_1*a*c**2*q_c/2 - I_2*M_a*c - I_2*V_a*c**2/2 + I_2*a**3*q_c/6 - I_2*a**2*c*q_c/2 + I_2*a*c**2*q_c/2 + I_2*c**3*q_m/6)/I_1 + (-F_br*I_1*b*c**2/2 + F_br*I_1*c**3/3 - F_br*I_2*b**3/6 + F_br*I_2*b*c**2/2 - F_br*I_2*c**3/3 - I_1*M_a*c**2/2 - I_1*V_a*c**3/3 - I_1*a**2*c**2*q_c/4 + I_1*a*c**3*q_c/3 + I_1*c**4*q_m/12 + I_2*M_a*c**2/2 + I_2*V_a*c**3/3 - I_2*a**4*q_c/24 + I_2*a**2*c**2*q_c/4 - I_2*a*c**3*q_c/3 - I_2*c**4*q_m/8)/I_1)/(E_1*I_2)',
            'M_xI': '-M_a - V_a*x + q_c*x**2/2 + q_m*x**2/2',
            'M_xII': '-M_a - V_a*x - a**2*q_c/2 + a*q_c*x + q_m*x**2/2',
            'M_xIII': '-F_br*b + F_br*x - M_a - V_a*x - a**2*q_c/2 + a*q_c*x + q_m*x**2/2',
            'M_xIIII': '-F_br*b - M_a - a**2*q_c/2 - c**2*q_m/2 + x*(F_br - V_a + a*q_c + c*q_m)',
            'V_xI': '-V_a + q_c*x + q_m*x',
            'V_xII': '-V_a + a*q_c + q_m*x',
            'V_xIII': 'F_br - V_a + a*q_c + q_m*x',
            'V_xIIII': 'F_br - V_a + a*q_c + c*q_m',
            'M_a': '-F_br*b + F_s*d - M_s - a**2*q_c/2 - c**2*q_m/2',
            'V_a': 'F_br - F_s + a*q_c + c*q_m',
        }

    def update(self, loads: ICylinderLoads):
        self._loads = loads

    def get_deflection(self, x: (float, np.ndarray)):
        return self.deflection(x)

    def get_rotation(self, x: (float, np.ndarray)):
        return self.rotation(x)

    def _update_variable_dict(self, x=None):
        """ Update values of internal variable dictionary

        @param x: value for dependent variable 'x'
        @return:
        """

        # Create load_dict:
        self._val_dict = dict({
            'a': self._layout.max_axial_line / 2,  # mm
            'b': self._cylinder.br_location,  # mm
            'c': self._cylinder.body.length / 2,  # mm
            'd': self._cylinder.body.length / 2 + self._bb.s,  # mm
            'q_c': self._loads.get_qcut(),  # N/mm
            'q_m': self._loads.get_qm(),
            'F_br': self._loads.get_Fbr(),  # N
            'F_s': self._loads.get_Fbearing(),  # N
            'E_1': self._cylinder.body.material.E,
            'I_1': self._cylinder.body.second_area_moment,
            'I_2': self._cylinder.os_shaft.second_area_moment,
            'x': x,
            'M_s': self._loads.get_Mshaft()
        })

        # Update equation-dependent values:
        self._val_dict['V_a'] = self._eval_expr('V_a', update_dict=False)
        self._val_dict['M_a'] = self._eval_expr('M_a', update_dict=False)

    def _sectioned_evaluation(self, x, key):
        """ Evaluate key expression at x while taking account for different cylinder_properties sections

        @param x: value for variable x
        @param key: Key of equation to evaluate
        @return: evaluation result
        """

        single_value = False
        if (type(x) is float) or (type(x) is int):
            x = np.array([x])
            single_value = True

        a = self._layout.max_axial_line / 2,  # mm
        b = self._cylinder.br_location,  # mm
        c = self._cylinder.body.length / 2,  # mm
        d = self._cylinder.body.length / 2 + self._bb.s,  # mm

        ind_0a = np.logical_and(0 <= x, x < a)
        ind_ab = np.logical_and(a <= x, x < b)
        ind_bc = np.logical_and(b <= x, x <= c)
        ind_cd = np.logical_and(c <= x, x <= d)

        _val = np.zeros(x.shape[0])
        _val[ind_0a] = self._eval_expr('{0}_xI'.format(key), x[ind_0a])
        _val[ind_ab] = self._eval_expr('{0}_xII'.format(key), x[ind_ab])
        _val[ind_bc] = self._eval_expr('{0}_xIII'.format(key), x[ind_bc])
        _val[ind_cd] = self._eval_expr('{0}_xIIII'.format(key), x[ind_cd])

        if single_value:
            _val = _val[-1]

        return _val

    def _eval_expr(self, variable_key, x=None, update_dict=True):
        """

        @param variable_key: Key-name of expression to evaluate
        @param x: value for dependent variable (float or ndarray)
        @param update_dict: flag to indicate if internal dictionary needs to be updated
        @return: result of expression evaluation
        """

        # Update dict
        if update_dict:
            self._update_variable_dict(x)

        # Make values available in local workspace:
        for name, value in self._val_dict.items():
            if value is not None:
                exec('{0} = self._val_dict[\'{0}\']'.format(name))

        # Return value
        res = eval('{0}'.format(self._eq_dict[variable_key]))
        return res

    @property
    def V_a(self):
        """Transversal force V_a [N]

                             __q_c + q_m__     F_br
                            |    |   |   |
                         |  v    v   v  _v
                         |----^----^----|  |
              ->      ^  |              |  |__|o|__
            /     V_a |  |              |  |__   __|
        M_a \_        |  |              |  |  |o|
                         |----v----v----|__|
                         |
        """
        return self._eval_expr('V_a')

    @property
    def M_a(self):
        """Internal Moment M_a [Nmm]

                          __q_c + q_m__     F_br
                             |    |   |   |
                         |   v    v   v  _v
                         |----^----^----|  |
              ->      ^  |              |  |__|o|__
            /     V_a |  |              |  |__   __|
        M_a \_        |  |              |  |  |o|
                         |----v----v----|__|
                         |
        """

        return self._eval_expr('M_a')

    def rotation(self, x):
        """ Cylinder rotation at x [rad]

                      __q_c + q_m__     F_br
                         |    |   |   |
                     |   v    v   v  _v
                     |----^----^----|  |
          ->      ^  |              |  |__|o|__
        /     V_a |  |              |  |__   __|
    M_a \_        |  |              |  |  |o|
                     |----v----v----|__|
                     |
                     |----------> x
        @param x: location(s) at which rotation should be evaluated
        """
        return self._sectioned_evaluation(x, key='phi')

    def deflection(self, x):
        """ Cylinder deflection at x[mm]

                      __q_c + q_m__     F_br
                         |    |   |   |
                     |   v    v   v  _v
                     |----^----^----|  |
          ->      ^  |              |  |__|o|__
        /     V_a |  |              |  |__   __|
    M_a \_        |  |              |  |  |o|
                     |----v----v----|__|
                     |
                     |----------> x

        @param x: location(s) at which rotation should be evaluated
        """
        return self._sectioned_evaluation(x, key='v')

    def bending_moment(self, x):
        """ Cylinder bending moment  at x[N*mm]

                      __q_c + q_m__     F_br
                         |    |   |   |
                     |   v    v   v  _v
                     |----^----^----|  |
          ->      ^  |              |  |__|o|__
        /     V_a |  |              |  |__   __|
    M_a \_        |  |              |  |  |o|
                     |----v----v----|__|
                     |
                     |----------> x

        @param x: location(s) at which rotation should be evaluated
        """
        return self._sectioned_evaluation(x, key='M')

    def shear_force(self, x):
        """ Shear force at x [N]
                      __q_c + q_m__     F_br
                         |    |   |   |
                     |   v    v   v  _v
                     |----^----^----|  |
          ->      ^  |              |  |__|o|__
        /     V_a |  |              |  |__   __|
    M_a \_        |  |              |  |  |o|
                     |----v----v----|__|
                     |
                     |----------> x

        @param x:
        @return:
        """
        return self._sectioned_evaluation(x, key='V')


class BobstToolsetLoads(IToolsetLoads):
    _upper_cylinder = ICylinder()
    _lower_cylinder = ICylinder()
    _upper_bearing_block = IBearingBlock()
    _lower_bearing_block = IBearingBlock()
    _layout = IProductLayout()
    F_t = CFloat()
    q_cut = CFloat()
    g = CFloat()

    def __init__(self, upper_cylinder: ICylinder, lower_cylinder: ICylinder,
                 upper_bearing_block: BearingBlock,
                 lower_bearing_block: BearingBlock,
                 layout: IProductLayout,
                 F_t: float, q_cut: float, g=9.81):
        """
        @param upper_cylinder: Upper cylinder
        @param lower_cylinder: lower cylinder
        @param bearing_block: bearing block
        @param layout: Toolset layout
        @param F_t: [N] Toolset tension
        @param q_cut: [N/mm] Cutting force
                          __                            __
                         |  |---^----^----^---^----^---|  |
             ->    __|o|__|  |                    ^     |  |__|o|__    <-
     Mext  /      |__   __|  | Cylinder Body   (d_body) |  |__   __|      \   Mext
           \        |o|  |  |                    v     |  |  |o|         /
                     |   |__|---v----v----v---v----v---|__|   |
                     v    ^     ^   ^   ^   ^   ^  ^    ^     v
                          |     |___|___|___|___|__|    |

                   F_t  F_br          q_cut           F_br  F_t
                                ____________________
                          |     |   |   |   |   |  |     |
                     ^    v_    v   v   v   v   v  v    _v    ^
                     |   |  |---^----^----^---^----^---|  |   |
                  __|o|__|  |                    ^     |  |__|o|__
      Mext /     |__   __|  | Cylinder Body   (d_body) |  |__   __|      \  Mext
           \->      |o|  |  |                    v     |  |  |o|         /
                     ^   |__|---v----v----v---v----v---|__|   ^       <-
                     |                                        |
                    F_sup                                   F_sup

        """
        self._lower_loads = None
        self._upper_loads = None
        super().__init__(
                         _upper_cylinder=upper_cylinder,
                         _lower_cylinder=lower_cylinder,
                         _upper_bearing_block=upper_bearing_block,
                         _lower_bearing_block=lower_bearing_block,
                         _layout=layout,
                         F_t=F_t,
                         q_cut=q_cut,
                         g=g
                         )

        self._upper_loads = SimpleCylinderLoads(F_bearing=-self.F_t,
                                   F_br=-self.F_br,
                                   q_cut=-self.q_cut,
                                   q_m=self._upper_cylinder.mass * self.g / self._upper_cylinder.body.length,
                                   M_shaft=0)

        self._lower_loads = SimpleCylinderLoads(F_bearing=self.F_t + 0.5 * (self._upper_cylinder.mass
                                                               + self._lower_cylinder.mass) * self.g,
                                               F_br=self.F_br,
                                               q_cut=self.q_cut,
                                               q_m=self._upper_cylinder.mass * self.g / self._upper_cylinder.body.length,
                                               M_shaft=0)


    @traitlets.observe('F_t', 'q_cut', '_upper_cylinder', '_lower_cylinder')
    def cb_Ft_change(self, change):
        if not (self._upper_loads is None or self._lower_loads is None):
            # Update loads:
            self._upper_loads.q_cut = -self.q_cut
            self._lower_loads.q_cut = self.q_cut

            self._upper_loads.q_m = self._upper_cylinder.mass * self.g / self._upper_cylinder.body.length
            self._lower_loads.q_m = self._lower_cylinder.mass * self.g / self._lower_cylinder.body.length

            self._upper_loads.F_bearing = -self.F_t
            self._lower_loads.F_bearing = self.F_t + 0.5 * (self._upper_cylinder.mass
                                                            + self._lower_cylinder.mass) * self.g

            self._upper_loads.F_br = -self.F_br
            self._lower_loads.F_br = self.F_br

            self._upper_cylinder.bearer_ring.load = self.upper_cylinder_loads.get_Fbearing()
            self._lower_cylinder.bearer_ring.load = self.lower_cylinder_loads.get_Fbearing()

            self._upper_bearing_block.bearing.load = self._upper_loads.F_bearing
            self._lower_bearing_block.bearing.load = self._lower_loads.F_bearing

    @property
    def upper_cylinder_loads(self) -> ICylinderLoads:
        return self._upper_loads

    @property
    def F_br(self):
        """ Bearer ring load

        @return:
        """
        return self.F_t + self._upper_cylinder.mass * self.g / 2 - self.q_cut * self._layout.max_axial_line / 2

    @property
    def lower_cylinder_loads(self) -> ICylinderLoads:
        return self._lower_loads

    def update(self, deflections: IToolsetDeflections):
        pass


class ToolsetDeflections(IToolsetDeflections):

    def __init__(self, toolset_loads: IToolsetLoads, upper_cylinder: ICylinder, lower_cylinder: ICylinder,
                 bearing_block: IBearingBlock, layout: IProductLayout):

        self._loads = toolset_loads
        self._upper_cylinder = upper_cylinder
        self._lower_cylinder = lower_cylinder
        self._bb = bearing_block

        self._upper_defl = CylinderDeflectionModel(cylinder=upper_cylinder,
                                                   bearing_block=bearing_block,
                                                   loads=toolset_loads.upper_cylinder_loads,
                                                   layout=layout)

        self._lower_defl = CylinderDeflectionModel(cylinder=lower_cylinder,
                                                   bearing_block=bearing_block,
                                                   loads=toolset_loads.lower_cylinder_loads,
                                                   layout=layout)

        self._bearer_contact = BearerRingContact(upper_cylinder.bearer_ring, lower_cylinder.bearer_ring)

    @property
    def upper_cylinder(self) -> ICylinderDeflections:
        return self._upper_defl

    @property
    def lower_cylinder(self) -> ICylinderDeflections:
        return self._lower_defl

    @property
    def bearer_deflection(self) -> float:
        return self._bearer_contact.indentation(self._loads.lower_cylinder_loads.get_Fbr())

    @property
    def bearer_contact(self) -> BearerRingContact:
        # Update load:
        self._bearer_contact.load = self._loads.lower_cylinder_loads.get_Fbr()
        return self._bearer_contact

    def gap_deflection(self, x) -> float:
        x_up = self.upper_cylinder.get_deflection(x)
        x_low = self.lower_cylinder.get_deflection(x)

        return x_up - x_low


class BobstToolset(IToolset):
    upper_cylinder = ICylinder()
    lower_cylinder = ICylinder()
    spacer = SpacerElement()
    tensioner = TensionElement()
    layout = IProductLayout()
    upper_gear = Gear(width=50, d_pitch=100, d_bore=50, material=steel, module=2)
    lower_gear = Gear(width=50, d_pitch=100, d_bore=50, material=steel, module=2)
    upper_bearing_block = IBearingBlock()
    lower_bearing_block = IBearingBlock()
    name = CUnicode()
    F_t = CFloat()
    q_cut = CFloat()

    def __init__(self,
                 upper_cylinder: ICylinder,
                 lower_cylinder: ICylinder,
                 spacer: SpacerElement,
                 tensioner: TensionElement,
                 layout: IProductLayout,
                 upper_gear: Gear,
                 lower_gear: Gear,
                 upper_bearing_block: IBearingBlock,
                 lower_bearing_block: IBearingBlock,
                 name: str,
                 F_t=20e3,
                 q_cut=20):
        self._load_model = None
        self._deflections = None

        mbb.TraitsXMLSerializer.__init__(self, upper_cylinder=upper_cylinder, lower_cylinder=lower_cylinder,
                                         spacer=spacer, tensioner=tensioner, layout=layout,
                                         upper_gear=upper_gear, lower_gear=lower_gear,
                                         upper_bearing_block=upper_bearing_block,
                                         lower_bearing_block=lower_bearing_block,
                                         name=name, F_t=F_t, q_cut=q_cut
                                         )

        self._load_model = BobstToolsetLoads(upper_cylinder=self.upper_cylinder,
                                             lower_cylinder=self.lower_cylinder,
                                             upper_bearing_block=self.upper_bearing_block,
                                             lower_bearing_block=self.lower_bearing_block,
                                             layout=self.layout,
                                             F_t=self.F_t,
                                             q_cut=self.q_cut
                                             )

        self._deflections = ToolsetDeflections(toolset_loads=self._load_model,
                                               upper_cylinder=self.upper_cylinder,
                                               lower_cylinder=self.lower_cylinder,
                                               bearing_block=self.upper_bearing_block,
                                               layout=self.layout
                                               )

        # Set force to ensure all bodies are updated
        self._update_loads()

    def _update_loads(self):
        self._load_model.F_t = self.F_t
        self._load_model.q_cut = self.q_cut

        self.bearer_contact.load = self._load_model.F_br

        self.upper_bearing_block.bearing.load = self._load_model.upper_cylinder_loads.F_bearing
        self.lower_bearing_block.bearing.load = self._load_model.lower_cylinder_loads.F_bearing

    @traitlets.observe('F_t', 'q_cut')
    def _cb_load_change(self, o):
        if self._load_model is not None:
            self._update_loads()

    @property
    def deflections(self) -> IToolsetDeflections:
        return self._deflections

    @property
    def loads(self) -> IToolsetLoads:
        return self._load_model

    @property
    def gap_stiffness(self) -> float:
        raise NotImplementedError()

    @property
    def bearer_contact(self):
        return self._deflections.bearer_contact

    @property
    def k_shaft_upper(self):
        # Todo: shaft stiffness is altered by deflection in response to gravity, this makes this estimate a bit off
        dx_br = self.deflections.upper_cylinder.get_deflection(self.upper_cylinder.br_location)
        dx_b = self.deflections.upper_cylinder.get_deflection(self.upper_cylinder.body.length / 2 + self.upper_bearing_block.s)
        return abs(self.loads.upper_cylinder_loads.get_Fbearing() / (dx_br - dx_b))

    @property
    def k_shaft_lower(self):
        # Todo: shaft stiffness is altered by deflection in response to gravity, this makes this estimate a bit off
        dx_br = self.deflections.lower_cylinder.get_deflection(self.lower_cylinder.br_location)
        dx_b = self.deflections.lower_cylinder.get_deflection(self.lower_cylinder.body.length / 2 + self.lower_bearing_block.s)
        return abs(self.loads.lower_cylinder_loads.get_Fbearing() / (dx_br - dx_b))

    def generate_dynamics(self):
        # Define bodies in toolset:
        bodies = {'bb_uOS': self.upper_bearing_block, 'bb_uDS': copy(self.upper_bearing_block),
                  'bb_lOS': self.lower_bearing_block, 'bb_lDS': copy(self.lower_bearing_block),
                  'dum_lOS': DummyBody(mass=1.0, subscript='d'), 'dum_lDS': DummyBody(mass=1.0, subscript='d'),
                  'dum_uOS': DummyBody(mass=1.0, subscript='d'), 'dum_uDS': DummyBody(mass=1.0, subscript='d'),
                  'gear_up': self.upper_gear, 'gear_lower': self.lower_gear,
                  'cyl_up': self.upper_cylinder, 'cyl_low': self.lower_cylinder,
                  }

        # Define spring connections
        f_ground = lambda b1, b2: b2.symbolic_states[0]
        f_trans = lambda b1, b2: b2.symbolic_states[0] - b1.symbolic_states[0]
        f_bb_cyl_OS = lambda bb, cyl: (
                (cyl.symbolic_states[0]
                 - symp.sin(cyl.symbolic_states[1]) * (cyl.body.length / 2 + self.upper_bearing_block.s) * 1e-3 # Convert to mm -> m
                 )
                -
                bb.symbolic_states[0]
        )
        f_bb_cyl_DS = lambda bb, cyl: (
                (cyl.symbolic_states[0]
                 + symp.sin(cyl.symbolic_states[1]) * (cyl.body.length / 2 + self.upper_bearing_block.s) * 1e-3  # Convert to mm -> m
                 )
                -
                bb.symbolic_states[0]
        )
        f_cyl_cylOS = lambda cyl1, cyl2: ((cyl2.symbolic_states[0]
                                           - symp.sin(cyl2.symbolic_states[1]) * (
                                                   cyl2.body.length / 2 + self.upper_bearing_block.s) * 1e-3
                                           )
                                          - (cyl1.symbolic_states[0]
                                             - symp.sin(cyl1.symbolic_states[1]) * (
                                                     cyl1.body.length / 2 + self.upper_bearing_block.s) * 1e-3
                                             )
                                          )

        f_cyl_cylDS = lambda cyl1, cyl2: ((cyl2.symbolic_states[0]
                                           + symp.sin(cyl2.symbolic_states[1]) * (
                                                   cyl2.body.length / 2 + self.upper_bearing_block.s) * 1e-3
                                           )
                                          -
                                          (cyl1.symbolic_states[0]
                                           + symp.sin(cyl1.symbolic_states[1]) * (
                                                   cyl1.body.length / 2 + self.upper_bearing_block.s) * 1e-3
                                           )
                                          )

        connections = {
            # OS connections
            'g->bbl_OS': mmech.BodyConnection(body1=Ground(), body2=bodies['bb_lOS'],
                                              connection_element=DummyStiffness(1e9),
                                              relation=f_ground),
            'bb_lOS->bbu_OS': mmech.BodyConnection(body1=bodies['bb_lOS'], body2=bodies['bb_uOS'],
                                                   connection_element=self.spacer,
                                                   relation=f_trans),
            'bb_lOS->dum_lOS': mmech.BodyConnection(body1=bodies['bb_lOS'], body2=bodies['dum_lOS'],
                                                    connection_element=self.lower_bearing_block.bearing,
                                                    relation=f_trans),
            'bb_uOS->dum_uOS': mmech.BodyConnection(body1=bodies['bb_uOS'], body2=bodies['dum_uOS'],
                                                    connection_element=self.upper_bearing_block.bearing,
                                                    relation=f_trans),
            'dum_lOS->cyl_low': mmech.BodyConnection(body1=bodies['dum_lOS'], body2=bodies['cyl_low'],
                                                     connection_element=DummyStiffness(self.k_shaft_lower),
                                                     relation=f_bb_cyl_OS),
            'dum_uOS->cyl_up': mmech.BodyConnection(body1=bodies['dum_uOS'], body2=bodies['cyl_up'],
                                                    connection_element=DummyStiffness(self.k_shaft_upper),
                                                    relation=f_bb_cyl_OS),
            'cyl_low->cyl_up_OS': mmech.BodyConnection(body1=bodies['cyl_low'], body2=bodies['cyl_up'],
                                                       connection_element=self.bearer_contact,
                                                       relation=f_cyl_cylOS),
            # DS connections
            'g->bbl_DS': mmech.BodyConnection(body1=Ground(), body2=bodies['bb_lDS'],
                                              connection_element=DummyStiffness(1e9),
                                              relation=f_ground),
            'bb_lDS->bbu_DS': mmech.BodyConnection(body1=bodies['bb_lDS'], body2=bodies['bb_uDS'],
                                                   connection_element=self.spacer,
                                                   relation=f_trans),
            'bb_lDS->dum_lDS': mmech.BodyConnection(body1=bodies['bb_lDS'], body2=bodies['dum_lDS'],
                                                    connection_element=self.lower_bearing_block.bearing,
                                                    relation=f_trans),
            'bb_uDS->dum_uDS': mmech.BodyConnection(body1=bodies['bb_uDS'], body2=bodies['dum_uDS'],
                                                    connection_element=self.upper_bearing_block.bearing,
                                                    relation=f_trans),
            'dum_lDS->cyl_low': mmech.BodyConnection(body1=bodies['dum_lDS'], body2=bodies['cyl_low'],
                                                     connection_element=DummyStiffness(self.k_shaft_lower),
                                                     relation=f_bb_cyl_DS),
            'dum_uDS->cyl_up': mmech.BodyConnection(body1=bodies['dum_uDS'], body2=bodies['cyl_up'],
                                                    connection_element=DummyStiffness(self.k_shaft_upper),
                                                    relation=f_bb_cyl_DS
                                                    ),
            'cyl_low->cyl_up_DS': mmech.BodyConnection(body1=bodies['cyl_low'], body2=bodies['cyl_up'],
                                                       connection_element=self.bearer_contact,
                                                       relation=f_cyl_cylDS
                                                       ),
            'dum_low->gearu': mmech.BodyConnection(body1=bodies['dum_uDS'], body2=bodies['gear_up'],
                                                   connection_element=DummyStiffness(self.k_shaft_upper),
                                                   relation=f_trans),
            'dum_low->gearl': mmech.BodyConnection(body1=bodies['dum_lDS'], body2=bodies['gear_lower'],
                                                   connection_element=DummyStiffness(self.k_shaft_lower),
                                                   relation=f_trans),
        }

        self._dynamics = SystemDynamics(bodies=bodies, body_connections=connections)

        return self._dynamics


class ConvexityComputer(object):
    def __init__(self, toolset: IToolset, side_mid_deviation: float):
        """ Computer for the cylinder_properties diameter variation

        @param toolset: Toolset for which to compute the variation
        @param side_mid_deviation: [mm] desired gap-variation between toolset center and measurement location
        """
        ts = deepcopy(toolset)

        self._toolset = ts

        self._side_mid_deviation = side_mid_deviation

    def compute(self, x_val=None):
        """
        Returns the diameter variation required to counter gap-variation due to toolset deflection.
        @param x_val: [mm] The distance from the cylinder_properties center at which the gap-variation should
        be computed.
        @return : [mm]
        """

        # Location at which to evaluate the deflection
        x_val = mbb.ArgumentVerifier((float, int), self._toolset.layout.max_axial_line / 2).verify(x_val)

        # Compute deflection of upper and lower
        gap_var = self._toolset.deflections.gap_deflection(x_val)

        # Convexity defined as diameter variation:
        return 2 * (gap_var + self._side_mid_deviation)


if __name__ == "__main__":
    # Load model
    MadernModelLibrary().load('../data/library/', verbose=True)

    # Material:

    print('Generating Material')
    steel = Material(rho=7.8e3, E=200e3, v=0.27, material_name='Steel')
    steel_xml = ET.ElementTree(steel.to_xml())
    steel_xml.write('steel.xml')
    print(steel)

    # SpacerElement
    M30x2 = Thread(d=30, d_pitch=29.026, d_root=27.546, material=steel)
    set_screw = SetScrew(height=8, diameter=27, thread=M30x2, material=steel)
    my_spacer = SpacerNSetScrews(set_screw, 2)

    # Tie-rod
    tie_rod = TensionRod(l_rod=500, d_rod=20, l_tube=400, d_itube=25, d_otube=30, thread=M30x2, material=steel)
    tie_rodx2 = TensionNTieRods(tie_rod, n=2)

    bearing = TaperedBearing(material=steel, r_inner=80, r_outer=100, r_roller=20, contact_length=10,
                             n_rollers=20, body_angle=5)
    Obearing = OTaperedBearing(bearing, effective_distance=20)

    bearer_ring = BearerRing(width=55.0, diameter=220.0, angle=1.0, material=steel, thickness=50.0)
    shaft = Shaft(length=200, outer_diameter=80, inner_diameter=0.0, material=steel)
    body = Shaft(length=788, outer_diameter=220, inner_diameter=0.0, material=steel)

    cyl = Cylinder(os_shaft=shaft, ds_shaft=shaft, body=body, bearer_ring=bearer_ring, br_location=350)
    bb = BearingBlock(Obearing, s=40, mass=60)
    layout = SimpleLayout(600)

    ts = BobstToolset(upper_cylinder=cyl, lower_cylinder=cyl,
                      spacer=my_spacer, tensioner=tie_rodx2, layout=layout,
                      lower_gear=Gear(width=50, d_pitch=220, d_bore=80, material=steel, module=2),
                      upper_gear=Gear(width=50, d_pitch=220, d_bore=80, material=steel, module=2),
                      lower_bearing_block=bb,
                      upper_bearing_block=copy(bb),
                      name="FakeBobst", F_t=20e3, q_cut=0)

    # Test XML read/write
    ts_xml = ET.ElementTree(ts.to_xml())
    ts_xml.write('./ts_test.xml')

    ts = BobstToolset.from_xml(ts_xml.getroot(), class_factory=ToolsetClassFactory())

    # Display loads
    print('Upper: \n{0}'.format(ts.loads.upper_cylinder_loads))
    print('Lower: \n{0}'.format(ts.loads.lower_cylinder_loads))

    # Convexity model
    print('--- Deflection Model ---')
    print('Upper        : {0:>7.2f} mu'.format(
        ts.deflections.upper_cylinder.get_deflection(x=ts.layout.max_axial_line / 2) * 1e3))
    print('Lower        : {0:>7.2f} mu'.format(
        ts.deflections.lower_cylinder.get_deflection(x=ts.layout.max_axial_line / 2) * 1e3))
    print('----------------------')
    print('Gap variation: {0:>7.2f} mu'.format(ts.deflections.gap_deflection(x=ts.layout.max_axial_line / 2) * 1e3))
    print(' ')
    print('---- Bearer deflection ----')
    print('dx : {0:.3f} mm'.format(ts.deflections.bearer_deflection))

    print('-----Convexity Info-')
    cc = ConvexityComputer(ts, side_mid_deviation=0)
    print('Male convexity: {0:>7.2f} mu'.format(cc.compute(x_val=ts.layout.max_axial_line / 2) * 1e3))

    print('-----Generate Dynamics-')
    dyn = ts.generate_dynamics()
    dyn.mode_shapes(dyn.value_dict)

    print('Finished')
