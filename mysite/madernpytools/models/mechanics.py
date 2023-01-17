import numpy as np
import traitlets
from traitlets import CFloat, CUnicode
import xml.etree.cElementTree as ET

import madernpytools.backbone as mbb
import sympy as symp


class MechanicsClassFactory(mbb.IClassFactory):

    def get(self, name):
        return eval(name)


class IEnergy(object):

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        raise NotImplementedError()


class IKineticEnergy(IEnergy):

    @property
    def symbolic_states(self) -> list:
        "Symbolic expression"
        raise NotImplementedError()

    @property
    def symbolic_state_derivatives(self) -> list:
        "Symbolic expression"
        raise NotImplementedError()

    @property
    def symbolic_state_2th_derivatives(self) -> list:
        "Symbolic expression"
        raise NotImplementedError()

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        raise NotImplementedError()

    def get_state_value(self, key) -> float:
        "Dictionary containing mapping between "
        raise NotImplementedError()


class IPotentialEnergy(IEnergy):

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        raise NotImplementedError()


class BodyConnection(IEnergy):

    def __init__(self, body1: IKineticEnergy,
                 body2: IKineticEnergy,
                 connection_element: IPotentialEnergy,
                 relation = None):

        self._body1 = body1
        self._body2 = body2
        self._conn_elem = connection_element

        if relation is None:
            relation = lambda b1, b2: b2.symbolic_states[0] - b1.symbolic_states[0]
        self._relation = relation

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        return self._conn_elem.get_energy_equation(self._relation(self._body1, self._body2))


class HertzContact(traitlets.HasTraits, traitlets.TraitType):

    R_x1 = CFloat()
    R_x2 = CFloat()
    R_y1 = CFloat()
    R_y2 = CFloat()
    R_E1 = CFloat()
    R_E2 = CFloat()
    R_v1 = CFloat()
    R_v2 = CFloat()

    def __init__(self, R_x1, R_x2, R_y1, R_y2, E_1, E_2, v_1, v_2):
        """ Generic Hertz contact

        :param R_x1: [mm] Radius in x direction of body 1
        :param R_x2: [mm] Radius in x direction of body 2
        :param R_y1: [mm] Radius in y direction of body 1
        :param R_y2: [mm] Radius in y direction of body 2
        :param E_1:  [N/mm2] E1 Elasticity modulus of body 1
        :param E_2:  [N/mm2] E2 Elasticity modulus of body 2
        :param v_1:  [-] Body 1 material poisson ratio
        :param v_2:  [-] Body 2 material poisson ratio
        """
        super().__init__(R_x1=R_x1, R_x2=R_x1,
                         R_y1=R_y1, R_y2=R_y2,
                         E_1=E_1, E_2=E_2,
                         v_1=v_1, v_2=v_2)

        # Compute contact radius
        self._R = ((R_x1 ** -1 if (R_x1 != 0.0) else 0.0) +
                   (R_x2 ** -1 if (R_x2 != 0.0) else 0.0) +
                   (R_y1 ** -1 if (R_y1 != 0.0) else 0.0) +
                   (R_y2 ** -1 if (R_y2 != 0.0) else 0.0)
                   ) ** -1

        # Compute contact E-modulus
        self._E = ((1 - v_1 ** 2) / (2 * E_1) + (1 - v_2 ** 2) / (2 * E_2)) ** -1

    @property
    def Eprime(self):
        """Contact E-modulus [N/mm2]"""
        return self._E

    @property
    def Rprime(self):
        """Contact Radius [mm]"""
        return self._R

    def contact_radius(self, F):
        """ Contact radius

        :param F: Applied force [N]
        :return: Contact radius [mm]
        """
        return (3 * F * self._R / self._E) ** (1 / 3)

    def contact_area(self, F):
        """ Contact area

        :param F:  Applied force [N]
        :return: Contact area [mm2]
        """
        return np.pi * self.contact_radius(F) ** 2

    def indentation(self, F):
        """ Combined displacement of the bodies in contact  [mm]

        :param F:  Applied force [N]
        :return: displacement [mm]
        """

        return self.contact_radius(F) ** 2 / (2 * self._R)

    def stiffness(self, F):
        """Stiffness of the contact at force F[N].

        :param F: Applied force
        :return:
        """
        return self._E * (2 * self._R * self.indentation(F)) ** 0.5


class SphericalHertzContact(object):

    def __init__(self, R_1, R_2, E_1, E_2, v_1, v_2):
        """ Generic Hertz contact

        :param R_1: [mm] Radius of body 1
        :param R_2: [mm] Radius of body 2
        :param E_1:  [N/mm2] E1 Elasticity modulus of body 1
        :param E_2:  [N/mm2] E2 Elasticity modulus of body 2
        :param v_1:  [-] Body 1 material poisson ratio
        :param v_2:  [-] Body 2 material poisson ratio
        """

        self._contact = HertzContact(R_x1=R_1, R_x2=R_1, R_y1=R_2, R_y2=R_2,
                                     E_1=E_1, E_2=E_2,
                                     v_1=v_1, v_2=v_2)

    @property
    def Eprime(self):
        """Contact E-modulus [N/mm2]"""
        return self._contact.Eprime

    @property
    def Rprime(self):
        """Contact Radius [mm]"""
        return self._contact.Rprime

    def contact_radius(self, F):
        """ Contact radius

        :param F: Applied force [N]
        :return: Contact radius [mm]
        """
        self._contact.contact_radius(F)

    def contact_area(self, F):
        """ Contact area

        :param F:  Applied force [N]
        :return: Contact area [mm2]
        """
        self._contact.contact_area(F)

    def indentation(self, F):
        """ Combined displacement of the bodies in contact  [mm]

        :param F:  Applied force [N]
        :return: displacement [mm]
        """
        self._contact.indentation(F)

    def stiffness(self, F):
        """Stiffness of the contact at force F[N].

        :param F: Applied force
        :return:
        """
        self._contact.stiffness(F)


class CylindricHertzContact(object):

    def __init__(self, R_1, R_2, L, E_1, E_2, v_1, v_2):
        """ Hertz contact for cylindric contact

        :param R_1: [mm] Radius of body 1
        :param R_2: [mm] Radius of body 2
        :param L  : [mm] Contact length
        :param E_1: [N/mm2] E1 Elasticity modulus of body 1
        :param E_2: [N/mm2] E2 Elasticity modulus of body 2
        :param v_1: [-] Body 1 material poisson ratio
        :param v_2: [-] Body 2 material poisson ratio
        """
        self._contact = HertzContact(R_x1=R_1, R_x2=0, R_y1=R_2, R_y2=0,
                                     E_1=E_1, E_2=E_2,
                                     v_1=v_1, v_2=v_2)
        self._R1 = R_1
        self._R2 = R_2
        self._E1 = E_1
        self._E2 = E_2
        self._v1 = v_1
        self._v2 = v_2
        self.L = L

    @property
    def Eprime(self):
        """Contact E-modulus [N/mm2]"""
        return self._contact.Eprime

    @property
    def Rprime(self):
        """Contact Radius [mm]"""
        return self._contact.Rprime

    def half_contact_width(self, F):
        """Half contact width b

        :param F: Force applied to the contact [N]
        :return:
        """
        E = self.Eprime
        R = self.Rprime
        L = self.L
        return np.sqrt(8 * F * R / (np.pi * L * E))

    def indentation(self, F):
        """Contact indentation

        :param F: Force applied to the contact [N]
        :return:
        """
        L = self.L
        R1 = self._R1
        E1 = self._E1
        v1 = self._v1

        R2 = self._R2
        E2 = self._E2
        v2 = self._v2

        b = self.half_contact_width(F)

        # [MZ] Added abs() around R1/R2 to account for negative curvature. second_area_moment'm not sure if this is correct
        delta = 2*F/(np.pi*L) * (((1-v1**2)/E1) * np.log(4*abs(R1)/b - 0.5) +
                                 ((1-v2**2)/E2) * np.log(4*abs(R2)/b - 0.5))
        return delta

    def stiffness(self, F):
        """Contact stiffness

        :param F: Applied force [N]
        :return:  Stiffness [N/mm]
        """
        delta = self.indentation(F)
        return F / delta


class IStiffnessElement(mbb.MadernObject, IPotentialEnergy):

    @property
    def k(self):
        """ Stiffness

        :return:
        """
        raise NotImplementedError()

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        raise NotImplementedError()


class IMaterial(mbb.TraitsXMLSerializer, mbb.MadernObject, traitlets.TraitType):

    info_text = 'Material definition'
    rho = CFloat(default_value=7850.0)
    E = CFloat(default_value=210e3)
    v = CFloat(default_value=0.3)
    eps = CFloat(default_value=6e-12)
    material_name = CUnicode('steel')


class Material(IMaterial):
    illustration_path = f''

    def __init__(self, rho, E, v, eps=12e-6, material_name=''):
        """

        :param rho: material density [kg/m3]
        :param E  : elasticity modulus [N/mm2]
        :param v  : poisson ratio [-]
        :param eps: Thermal expansion (1/K)
        """
        # XML serialization (assume variables so-far
        mbb.TraitsXMLSerializer.__init__(self, rho=rho, E=E, v=v, eps=eps, material_name=material_name)

    @property
    def name(self):
        return self.material_name

    @name.setter
    def name(self, value):
        self.material_name = value

    @traitlets.validate('rho', 'E', 'v')
    def _validate_items(self, proposal):
        """ Material density

        :return: density [kg/m3]
        """
        if proposal['value'] < 0:
            traitlets.TraitError('{0} value should be larger than 0.0'.format(proposal['name']))
        return proposal['value']

    def __str__(self):
        return '{3} (rho: {0} [kg/m3], e={1}[n/mm2], v={2})'.format(self.rho, self.E, self.v, self.material_name)

    def _validate(self, obj, value):
        """

        @param obj:
        @param value:
        @return:
        """

        if value.E > 0.0 and value.v > 0.0 and value.rho > 0.0:
            return value
        else:
            return traitlets.TraitError(obj, value)


steel = Material(rho=7.8e3, E=200e3, v=0.27, material_name='Steel')


class Thread(IStiffnessElement, mbb.TraitsXMLSerializer, traitlets.TraitType):
    illustration_path=''
    d = CFloat(default_value=.0)
    d_pitch = CFloat(default_value=0.0)
    d_root = CFloat(default_value=0.0)
    material = IMaterial()

    def __init__(self, d: float, d_pitch: float, d_root: float, material: IMaterial):
        """ Thread stiffness approximation

        :param        d: [mm] Nominal thread diameter
        :param  d_pitch: [mm] Pitch diameter (often denoted d_2)
        :param   d_root: [mm] Root diameter (often denoted d_3)
        """

        mbb.TraitsXMLSerializer.__init__(self, d=d, d_pitch=d_pitch, d_root=d_root, material=material)

    @property
    def k(self):
        """ Stiffness value

        :return: stiffness [N/mm]
        """
        l0 = self.d
        return self.material.E * self.A_t / l0

    @property
    def A_t(self):
        """ Tension surface area
        :return: Tension surface area [mm2]
        """
        # Todo: Verify formula
        d0 = (self.d_pitch + self.d_root)
        return np.pi / 4 * d0 ** 2

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        return 0.5 * self.k * state_eq ** 2


    #def _validate(self, obj, value):
    #    """
#
#        @param obj:
#        @param value:
#        @return:
#        """
#
#        if value.d_root < value.d_pitch < value.d:
#            return value
#
#        traitlets.TraitError(obj, value)

    def __str__(self):
        return f'M{self.d_pitch}'


class Tube(IStiffnessElement, mbb.TraitsXMLSerializer):

    L = CFloat(default_value=1.0)
    d_inner = CFloat(default_value=0.0)
    d_outer = CFloat(default_value=1.0)
    E = CFloat(default_value=210e3)

    def __init__(self, L: float, d_inner: float, d_outer: float, E: float):
        """

        :param L: [mm] Tube length
        :param d_inner: [mm] inner diameter
        :param d_outer: [mm] outer diameter
        :param E: [N/mm2] Material Elasticity module
        """
        # XMl serialization
        mbb.TraitsXMLSerializer.__init__(self, L=L, d_inner=d_inner, d_outer=d_outer, E=E)

    @traitlets.validate('L', 'E')
    def _validate_item(self, proposal):
        if proposal['value'] > 0.0:
            return proposal['value']
        raise traitlets.TraitError("{1} should be larger than {0}".format(0.0, proposal.trait.name))

    @traitlets.validate('d_inner')
    def _validate_dinner(self, proposal):
        if proposal['value'] < self.d_outer:
            return proposal['value']
        raise traitlets.TraitError("d_inner should be smaller than {0}".format(self.d_outer))

    @traitlets.validate('d_outer')
    def _validate_douter(self, proposal):
        if proposal['value'] > self.d_inner:
            return proposal['value']
        raise traitlets.TraitError("d_outer should be larger than {0}".format(self.d_inner))

    @property
    def A_t(self):
        """

        :return: Tension surface area [mm2]
        """
        return np.pi * (self.d_outer ** 2 - self.d_inner ** 2) / 4

    @property
    def k(self):
        """ Tube stiffness

        :return: [N/mm] stiffness
        """
        L0 = self.L
        return self.E * self.A_t / L0

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        return 0.5 * self.k * state_eq ** 2

    def _validate(self, obj, value):
        """

        @param obj:
        @param value:
        @return:
        """

        if value.d_inner < value.d_outer and value.L > 0.0 and value.E > 0.0:
            return value
        traitlets.TraitError(obj, value)


class Rod(IStiffnessElement, mbb.TraitsXMLSerializer, traitlets.TraitType):

    def __init__(self, L, d, E):
        """ Rod stiffness

        :param L: [mm] rod length
        :param d: [mm] diameter of rod
        :param E: [N/mm2] Elasticity module
        """
        mbb.TraitsXMLSerializer.__init__(self, )
        self._tube = Tube(L=L, d_inner=0.0, d_outer=d, E=E)

    @property
    def diameter(self):
        return self._tube.d_outer

    @property
    def length(self):
        return self._tube.L

    @property
    def A_t(self):
        return self._tube.A_t

    @property
    def k(self):
        return self._tube.k

    def to_xml(self):
        root = ET.Element('RodStiffness')
        root.set('Type', 'RodStiffness')
        root.append(self._tube.to_xml())
        return root

    @staticmethod
    def from_xml(xml_element, *args, **kwargs):
        tmp = Tube.from_xml(xml_element.find('TubeStiffness'), MechanicsClassFactory())
        return Rod(L=tmp.L, d=tmp.d_outer, E=tmp.E)

    def get_energy_equation(self, state_eq: symp.Expr) -> symp.Expr:
        "Energy expression of dynamic object"
        return 0.5 * self.k * state_eq ** 2

    def _validate(self, obj, value):
        """

        @param obj:
        @param value:
        @return:
        """
        return value


if __name__ == "__main__":

    def observer(o):
        print('Received:', o)

    steel = Material(rho=7.8e3, E=200e3, v=0.27, name='Steel')
    steel.observe(handler=observer)
    steel.E = 210e3
    thr = Thread(d=20, d_pitch=19, d_root=18, material=steel)
    tub = Tube(L=1, d_inner=10, d_outer=20, E=210e3)
    rod = Rod(L=10, d=20, E=210e3)


