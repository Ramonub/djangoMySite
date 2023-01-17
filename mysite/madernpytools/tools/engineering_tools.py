import madernpytools.backbone as mbb
from madernpytools.models.toolset_model import BearerRing, steel
import traitlets
import numpy as np


# Convexity computer

class BearerMountingConditions(mbb.MadernObject, mbb.TraitsXMLSerializer):
    dT = traitlets.CFloat(default_value=80)
    diametrical_clearance = traitlets.CFloat(default_value=0.08)
    fit_pressure = traitlets.CFloat(default_value=80)

    def __init__(self, dT, fit_pressure, diametrical_clearance):
        """

        @param dT: Temperature difference between bore and shaft parts
        @param fit_pressure: Desired interference fit pressure
        @param diametrical_clearance: Required clearance for installation
        """
        super().__init__(dT=dT, fit_pressure=fit_pressure, diametrical_clearance=diametrical_clearance)


class BearerPressFitDesign(mbb.MadernObject):
    bearer = BearerRing(diameter=300, width=50, angle=1, material=steel, thickness=40)
    mounting_conditions = BearerMountingConditions(dT=80, fit_pressure=18, diametrical_clearance=0.01)

    def __init__(self, bearer: BearerRing, mounting_conditions: BearerMountingConditions):
        super().__init__(bearer=bearer, mounting_conditions=mounting_conditions)

    @property
    def interference(self):
        """ Returns the press-fit diametrical interference

        """
        p = self.mounting_conditions.fit_pressure  # Fit pressure (MPa)
        d_o = self.bearer.diameter                 # Outer diameter (mm)
        d_in = self.bearer.nominal_bore_diameter   # Nominal Inner diameter (mm)
        E = self.bearer.material.E                 # MPa E-modulus

        d_od2 = (d_o / d_in) ** 2

        return p * (d_in / E) * ((d_od2 + 1) / (d_od2 - 1) + 1)

    @property
    def thermal_expansion(self):
        """ Returns the bearer thermal expansion during installation

        """
        d = self.bearer.nominal_bore_diameter
        eps = self.bearer.material.eps
        dT = self.mounting_conditions.dT

        return d * eps * dT

    @property
    def br_bore_press_fit_diameter(self):
        """ The bearer bore, required to esthablish the desired interference pressure
        @return:
        """
        return self.bearer.nominal_bore_diameter - self.interference

    @property
    def installation_clearance(self):
        """ Returns the clearance required for installation

        @return:
        """
        delta = self.interference
        dl_dT = self.thermal_expansion

        cl = dl_dT - delta

        return cl

    @property
    def meets_installation_requirements(self):
        return self.installation_clearance > self.mounting_conditions.diametrical_clearance


class ToolSetupParameters(mbb.MadernObject):
    s_init = traitlets.CFloat(default_value=1.0)
    s_sym = traitlets.CFloat(default_value=2.5)
    gap_tolerance = traitlets.CFloat(default_value=0.01)

    def __init__(self, s_init, s_sym, gap_tolerance):
        super().__init__(s_init=s_init, s_sym=s_sym, gap_tolerance=gap_tolerance)


class BearerGrindingDiameters(mbb.MadernObject):
    male_bearer = BearerRing(diameter=300, width=50, angle=1, material=steel, thickness=40)
    female_bearer = BearerRing(diameter=300, width=60, angle=1, material=steel, thickness=40)
    tool_setup_parameters = ToolSetupParameters(s_init=1.0, s_sym=2.5, gap_tolerance=0.01)

    def __init__(self, male_bearer: BearerRing, female_bearer: BearerRing, tool_setup_parameters: ToolSetupParameters):
        """

        @param male_bearer:
        @param female_bearer:
        @param s_init:
        @param s_sym:
        """
        super().__init__(male_bearer=male_bearer, female_bearer=female_bearer,
                         tool_setup_parameters=tool_setup_parameters)


    @property
    def symmetric_slip_diameter(self):
        """ Returns the diameter at which 'zero' slip occurs when body<-> bearer distance is set to 's_sym'.

        @param d_m: (mm) Male image diameter
        @param d_f: (mm) Female image diameter
        @param s_sym:  (mm) body <-> bearer distance symmetric slip distribution range
        @param s_init:  (mm) body <-> bearer distance at assembly (initial setting)
        @param alpha: (deg) conical angle
        @param T: (mm) Required tolerance
        """
        d_m = self.male_bearer.diameter
        d_f = self.female_bearer.diameter
        T = self.tool_setup_parameters.gap_tolerance
        s_sym = self.tool_setup_parameters.s_sym
        s_init = self.tool_setup_parameters.s_init
        alpha = self.male_bearer.angle

        return 0.5 * (d_m + d_f + T) - (s_sym - s_init) * np.tan(alpha / 180 * np.pi)

    @property
    def brmale_max_diameter(self):
        """

        @param d_sym_slip:  (mm) Bearer ring diameter at 'symmetric-slip' condition
        @param w_br:  (mm) Male bearer width
        @param alpha: (deg) conical angle
        """
        w_br = self.male_bearer.width
        alpha = self.male_bearer.angle
        d_sym_slip = self.symmetric_slip_diameter

        return w_br * np.tan(alpha / 180 * np.pi) + d_sym_slip


    @property
    def brfemale_max_diameter(self):
        """


        @param d_sym_slip:  (mm) Bearer ring diameter at 'symmetric-slip' condition
        @param w_br:  (mm) Male bearer width
        @param s_sym:  (mm) body <-> bearer distance symmetric slip distribution range
        @param alpha: (deg) conical angle
        """
        w_br = self.male_bearer.width              # The max diameter of female depends on the male width!
        alpha = self.female_bearer.angle
        s_sym = self.tool_setup_parameters.s_sym
        s_init = self.tool_setup_parameters.s_init
        d_sym_slip = self.symmetric_slip_diameter

        return (w_br + 2 * s_sym ) * np.tan(alpha / 180 * np.pi) + d_sym_slip





#