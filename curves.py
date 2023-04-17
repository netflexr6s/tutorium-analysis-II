from manimlib import *
from scipy.integrate import odeint
from math import cos, sin


# Parametrisierungen
# 2D
circ = [lambda t: np.array([cos(t),sin(t),0]), (0,2*PI)]
doublecirc = [lambda t: np.array([cos(2*t),sin(2*t),0]), (0,2*PI)]
lemniscate = [
    lambda t: np.array([3*cos(t) / (1+sin(t)**2), 3*sin(t)*cos(t) / (1+sin(t)**2),0]),
    (0,2*PI)
    ]
test = [lambda t: np.array([t*cos(5*t),t*sin(abs(5*t)),0]),(-PI,PI)]

# 3D
helix = lambda t: np.array([cos(t),sin(t),np.sqrt(t)])
spherical1 = lambda t: np.array([sin(2*t)*cos(3*t),sin(2*t)*sin(3*t),cos(2*t)])
spherical2 = lambda t: np.array([sin(3*t)*cos(5*t),sin(3*t)*sin(5*t),cos(3*t)])


class TracedCurve(Scene):
    def construct(self):
        ax = NumberPlane()
        dot = GlowDot()
        self.dot = dot
        
        func = doublecirc

        self.func = func[0]
        self.range = func[1]
        self.rate_func = smooth
        
        curve = ParametricCurve(
            self.func,
            self.range
        )

        parameter_space = NumberLine(
            (np.floor(self.range[0]),np.ceil(self.range[1])),
            include_numbers = True
        ).set_color(WHITE).scale(0.75)
        parameter_space.to_corner(DR)

        parameter_dot = Dot().set_color(YELLOW).move_to(
            parameter_space.n2p(self.range[0])
        )
        parameter = DecimalNumber(
            0, font_size = 38, 
            num_decimal_places = 2
        )
        parameter.add_updater(
            lambda x: x.set_value(
                parameter_space.p2n(
                    parameter_dot.get_center()
                )
            )
        )
        t_label = Tex("t=", font_size = 38)
        par_label = VGroup(t_label, parameter).set_color(YELLOW)
        par_label.arrange(RIGHT, buff=SMALL_BUFF)
        always(par_label.next_to, parameter_dot, UP)

        parameter_path = Line(
            parameter_space.n2p(self.range[0]),
            parameter_space.n2p(self.range[1])
        )
        par = VGroup(
            parameter_space, parameter_dot, par_label
        )

        trace = TracedPath(
            lambda: dot.get_center(), 
            stroke_width = 3
        ).set_color(RED)

        dot.move_to(self.func(self.range[0]))
        self.add(ax, trace, dot)
        self.add(par)
        self.interact()

        self.play(
            MoveAlongPath(
                dot, curve,
                rate_func = self.rate_func 
            ),
            MoveAlongPath(
                parameter_dot, parameter_path
            ),
            run_time = 5
        )


    # Interaktion
        dot2 = GlowDot()
        dot2.add_updater(
            lambda x: x.move_to(
                self.func(
                    parameter_space.p2n(
                        parameter_dot.get_center()
                    )
                )
            )
        )
        self.remove(dot)
        self.add(dot2)

        self.search_set = VGroup(parameter_dot)
        self.line = parameter_space

    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.search_set)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point) 
        mob.add_updater(
            lambda x: x.set_x(
                self.mouse_drag_point.get_center()[0]
            ) 
        )
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.search_set.clear_updaters() 

    
    # Tangentialvektor
    def make_vector(self):
        pass



class TracedCurve3D(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera
    }
    def construct(self):
        plane = NumberPlane(
            (-3,3),
            (-3,3),
            width = 6,
            height = 6
        ).set_opacity(0.5)
        ax = ThreeDAxes(
            (-3,3),
            (-3,3),
            (-3,3),
            width = 6,
            height = 6
        )
        dot = GlowDot()

        self.func = spherical1
        self.range = (0,2*PI)

        trace = TracedPath(
            lambda: dot.get_center(), 
            stroke_width = 3
        ).set_color(RED)

        curve = ParametricCurve(
            self.func,
            self.range
        )

        parameter_space = NumberLine(
            (np.floor(self.range[0]),np.ceil(self.range[1])),
            include_numbers = True
        ).set_color(WHITE).scale(0.75)
        parameter_space.to_corner(DR)

        parameter_dot = Dot().set_color(YELLOW).move_to(parameter_space.n2p(0))
        parameter = DecimalNumber(
            0, font_size = 38, 
            num_decimal_places = 2
        )
        parameter.add_updater(
            lambda x: x.set_value(
                parameter_space.p2n(
                    parameter_dot.get_center()
                )
            )
        )
        t_label = Tex("t=", font_size = 38)
        par_label = VGroup(t_label, parameter).set_color(YELLOW)
        par_label.arrange(RIGHT, buff=SMALL_BUFF)
        always(par_label.next_to, parameter_dot, UP)

        parameter_path = Line(
            parameter_space.n2p(0),
            parameter_space.n2p(self.range[1])
        )
        par = VGroup(
            parameter_space, parameter_dot, par_label
        ).fix_in_frame()


        dot.move_to(self.func(0))
        self.add(plane, ax, dot, trace)
        self.add(par)
        self.interact()

        self.play(
            MoveAlongPath(
                dot, curve
            ),
            MoveAlongPath(
                parameter_dot, parameter_path
            ),
            run_time = 5
        )

    # Interaktion
        dot2 = GlowDot()
        dot2.add_updater(
            lambda x: x.move_to(
                self.func(
                    parameter_space.p2n(
                        parameter_dot.get_center()
                    )
                )
            )
        )
        self.remove(dot)
        self.add(dot2)

        self.search_set = VGroup(parameter_dot)
        self.line = parameter_space
        
    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.search_set)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point) 
        mob.add_updater(
            lambda x: x.set_x(
                self.mouse_drag_point.get_center()[0]
            ) 
        )
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.search_set.clear_updaters()

        

class GraphOfCurve(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera
    }
    def construct(self):
        self.range = (0,2*PI)
        ax = ThreeDAxes(
            (np.floor(self.range[0]),np.ceil(self.range[1])),
            (-3,3),
            (-3,3),
            width=6
        )
        self.func = lambda t: ax.c2p(t, 3*cos(t) / (1+sin(t)**2), 3*sin(t)*cos(t) / (1+sin(t)**2)) #ax.c2p(t, cos(t), sin(t))
        self.proj_func = lambda t: ax.c2p(0, 3*cos(t) / (1+sin(t)**2), 3*sin(t)*cos(t) / (1+sin(t)**2))
        dot = GlowDot()

        trace = TracedPath(
            lambda: dot.get_center(), 
            stroke_width = 3
        ).set_color(RED)

        curve = ParametricCurve(
            self.func,
            self.range
        )

        parameter_space = NumberLine(
            (np.floor(self.range[0]),np.ceil(self.range[1])),
            include_numbers = True
        ).set_color(WHITE).scale(0.75)
        parameter_space.to_corner(DR)
        parameter_dot = Dot().set_color(YELLOW).move_to(parameter_space.n2p(0))

        parameter_space2 = ax.get_axis(0)
        parameter_dot2 = Sphere(radius=0.1).set_color(YELLOW).move_to(parameter_space2.n2p(0))
        parameter = DecimalNumber(
            0, font_size = 38, 
            num_decimal_places = 2
        )
        
        parameter.add_updater(
            lambda x: x.set_value(
                parameter_space.p2n(
                    parameter_dot.get_center()
                )
            )
        )
        t_label = Tex("t=", font_size = 38)
        par_label = VGroup(t_label, parameter).set_color(YELLOW)
        par_label.arrange(RIGHT, buff=SMALL_BUFF)
        always(par_label.next_to, parameter_dot, UP)

        parameter_path = Line(
            parameter_space.n2p(0),
            parameter_space.n2p(self.range[1])
        )
        parameter_path2 = Line(
            ax.c2p(0,0,0),
            ax.c2p(self.range[1],0,0)
        )
        par = VGroup(
            parameter_space, parameter_dot, par_label
        ).fix_in_frame()

        parameter_plane = NumberPlane(  
            (-3,3), (-3,3)
        )
        parameter_plane.apply_matrix(
            [[0,0,1],
             [1,0,0],
             [0,1,0]]
        )
        parameter_plane.shift(
            ax.get_origin() - parameter_plane.get_origin()
        )
        parameter_plane2 = parameter_plane.copy()

        dot.move_to(self.func(0))
        self.add(ax, dot, trace)
        self.add(par, parameter_dot2)
        self.interact()
        self.play(
            Write(parameter_plane)
        )
        self.interact()
        self.play(
            MoveAlongPath(
                dot, curve
            ),
            MoveAlongPath(
                parameter_dot, parameter_path
            ),
            MoveAlongPath(
                parameter_dot2, parameter_path2
            ),
            run_time = 5
        )

    # Interaktion
        dot2 = GlowDot()
        dot2.add_updater(
            lambda x: x.move_to(
                self.func(
                    parameter_space.p2n(
                        parameter_dot.get_center()
                    )
                )
            )
        )
        self.remove(dot)
        self.add(dot2)

        parameter_dot2.add_updater(
            lambda x: x.move_to(
                    parameter_space2.n2p(
                        parameter_space.p2n(parameter_dot.get_center())
                    )
                )
            )
        self.interact()
        self.play(
            parameter_plane.animate.shift(
                parameter_dot2.get_center() - parameter_plane.get_origin()
            )
        )
        parameter_plane.add_updater(
            lambda x: x.shift(
                parameter_dot2.get_center() - parameter_plane.get_origin()
            )
        )
        self.search_set = Group(parameter_dot)
        self.interact()

        projection = GlowDot()
        projection.add_updater(
            lambda x: x.move_to(
                self.proj_func(
                    parameter_space.p2n(
                        parameter_dot.get_center()
                    )
                ) 
            )
        )
        proj_trace = TracedPath(
            lambda: projection.get_center(), 
            stroke_width = 3
        ).set_color(RED)

        self.play(
            Write(parameter_plane2)
        )
        self.add(projection, proj_trace)
        self.interact()
        self.play(
            Uncreate(parameter_plane),
            FadeOut(
                Group(parameter_dot2, trace, dot2)
            )
        )
        self.interact()


    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.search_set)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point) 
        mob.add_updater(
            lambda x: x.set_x(
                self.mouse_drag_point.get_center()[0]
            ) 
        )
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.search_set.clear_updaters()



class LorenzAttractor(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera
    }
    def construct(self):
        frame = self.camera.frame
        frame.set_euler_angles(phi=80*DEGREES,theta=35*DEGREES)  
        frame.scale(0.9)

        ax = ThreeDAxes()
        def lorenz(t, state, s=10, r=28, b=8/3):
            x, y, z = state
            x_dot = s*(y - x)
            y_dot = r*x - y - x*z
            z_dot = x*y - b*z
            return [x_dot, y_dot, z_dot]
        
        def lorenz_curve(init): 
            grid = np.arange(0.0,40.0,0.01)
            res = odeint(lorenz, init, grid, tfirst=True)
            p = VMobject()
            p.set_points_as_corners([*[[a,b,c] for a,b,c in zip(res[:,0],res[:,1],res[:,2])]])
            p.set_stroke(None,1)
            p.make_approximately_smooth().scale(0.1).move_to(ORIGIN)
            return p
        
        colors = [RED, ORANGE, YELLOW] 

        dot1 = GlowDot()
        dot2 = GlowDot()

        trace_tail1 = TracingTail(
            lambda: dot1.get_center(), 
            stroke_width = 3,
            time_traced = 2
        ).set_color_by_gradient(RED)

        trace_tail2 = TracingTail(
            lambda: dot2.get_center(), 
            stroke_width = 3,
            time_traced = 2
        ).set_color_by_gradient(BLUE)

        trace1 = TracedPath(
            lambda: dot1.get_center(), 
            stroke_width = 2,
        ).set_color_by_gradient(RED)

        trace2 = TracedPath(
            lambda: dot2.get_center(), 
            stroke_width = 2,
        ).set_color_by_gradient(BLUE)

        #self.add(trace_tail1, trace_tail2)
        self.add(ax)
        self.add(trace1, trace2)
        self.add(dot1, dot2)
        self.play(
            MoveAlongPath(
                dot1, lorenz_curve([0.0,1.0,1.05]),
                rate_func = linear
            ),
            MoveAlongPath(
                dot2, lorenz_curve([0.1,1.5,1.08]),
                rate_func = linear
            ),
            run_time = 40
        )
        self.wait(2)



class DoublePendulum(Scene):
    CONFIG = {
        "L1": 2,
        "L2": 1.5,
        "m1": 3,
        "m2": 4,
        "g": 9.8,
        "init_t1": 60*DEGREES,
        "init_t2": 1.5, #30*DEGREES,
        "init_o1": 2,
        "init_o2": 0,
    }
    def construct(self):
        L1 = self.L1
        L2 = self.L2

        bg = NumberPlane(
            axis_config = {
                "stroke_color": BLUE_E
            }
        )
        dot1 = Dot(L1*bg.c2p(sin(self.init_t1),-cos(self.init_t1))).set_color(YELLOW)
        dot2 = Dot(
            L1*bg.c2p(sin(self.init_t1),-cos(self.init_t1)) + L2*bg.c2p(sin(self.init_t2),-cos(self.init_t2))
        ).set_color(RED)

        line1 = always_redraw(
            lambda: Line(bg.c2p(0,0), dot1.get_center()).set_color(WHITE)
        )
        line2 = always_redraw(
            lambda: Line(dot1.get_center(), dot2.get_center()).set_color(WHITE)
        )

        p1 = self.get_angles([self.init_t1, self.init_o1, self.init_t2, self.init_o2])[:,0] 
        p2 = self.get_angles([self.init_t1, self.init_o1, self.init_t2, self.init_o2])[:,2]

        x1 = [L1*sin(t1) for t1 in p1]
        y1 = [-L1*cos(t1) for t1 in p1]

        x2 = [L2*sin(t2)+L1*sin(t1) for t1, t2 in zip(p1, p2)]
        y2 = [-L2*cos(t2)-L1*cos(t1) for t1, t2 in zip(p1, p2)]


        self.add(bg, line1, line2, dot1, dot2)
        trace = TracingTail(dot2, time_traced=2).set_stroke(RED, 1) 
        self.add(trace)
        self.interact()
        for k in range(len(x1)-1):
            self.play(
                dot1.animate.move_to(x1[k+1]*RIGHT + y1[k+1]*UP),
                dot2.animate.move_to(x2[k+1]*RIGHT + y2[k+1]*UP),
                run_time = 0.1,
                rate_func = linear
            )
        self.wait()

    
    def get_angles(self, pos):
        L1 = self.L1
        L2 = self.L2
        M1 = self.m1
        M2 = self.m2
        G = self.g
        
        def DGL(t, state):
            dydx = np.zeros_like(state)
            dydx[0] = state[1]
            delta = state[2] - state[0]
            den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
            dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                        + M2 * G * sin(state[2]) * cos(delta)
                        + M2 * L2 * state[3] * state[3] * sin(delta)
                        - (M1+M2) * G * sin(state[0]))
                       / den1)
            dydx[2] = state[3]
            den2 = (L2/L1) * den1
            dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                        + (M1+M2) * G * sin(state[0]) * cos(delta)
                        - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                        - (M1+M2) * G * sin(state[2]))
                        / den2)
            return dydx

        def path(init): 
            grid = np.arange(0.0, 30, 0.08)
            res = odeint(DGL, init, grid, tfirst=True)
            return res
        
        return path(pos)
    


class Bifurcation(Scene):
    CONFIG = {
        "camera_class": ThreeDCamera,
        "L1": 2,
        "L2": 2,
        "m1": 4,
        "m2": 4,
        "g": 9.8,
        "init_t1": PI / 6,
        "init_t2": PI / 6,
        "init_o1": 0,
        "init_o2": 0,
    }
    def construct(self):
        ax = ThreeDAxes((0,180), (-10, 10), (-10,10), width=10, height=7, depth=7)
        self.ax = ax
        self.add(ax)
        samples = [k*DEGREES for k in range(0,180)]
        for t in samples:
            self.add(
                self.get_dots(t)
            )


    def get_values(self, t1):
        L1 = self.L1
        L2 = self.L2
        M1 = self.m1
        M2 = self.m2
        G = self.g
        t2 = self.init_t2
        o1 = self.init_o1
        o2 = self.init_o2

        def DGL(t, state):
            dydx = np.zeros_like(state)
            dydx[0] = state[1]
            delta = state[2] - state[0]
            den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
            dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                        + M2 * G * sin(state[2]) * cos(delta)
                        + M2 * L2 * state[3] * state[3] * sin(delta)
                        - (M1+M2) * G * sin(state[0]))
                       / den1)
            dydx[2] = state[3]
            den2 = (L2/L1) * den1
            dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                        + (M1+M2) * G * sin(state[0]) * cos(delta)
                        - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                        - (M1+M2) * G * sin(state[2]))
                        / den2)
            return dydx

        def path(init): 
            grid = np.arange(0.0, 30, 0.08)   ## [t1, o1, t2, o2]
            res = odeint(DGL, init, grid, tfirst=True)
            return res
        
        liste = path([t1, o1, t2, o2])
        eps = 0.4
        ausgabe = [
            [t, o, o2] for t,o,o2 in zip(liste[:,0], liste[:,1], liste[:,3]) if -eps < t < eps
        ]
        return ausgabe
    
    def get_dots(self, t1):
        dots = [
            GlowDot(radius=2e-2, glow_factor=2).set_color(YELLOW).move_to(
                self.ax.c2p(t1*(1/DEGREES), coords[1], coords[2])
            )
            for coords in self.get_values(t1)
        ]
        return Group(*dots)
