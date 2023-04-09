from manimlib import *
from scipy.integrate import odeint


Lg_formula_config = {
    "tex_to_color_map": {
        "\\theta_0": WHITE,
        "{L}": BLUE,
        "{g}": YELLOW,
    },
}


def get_ode():
    tex_config = {
        "tex_to_color_map": {
            "{\\theta}": BLUE,
            "{\\dot\\theta}": RED,
            "{\\ddot\\theta}": YELLOW,
            "{t}": WHITE,
            "{\\mu}": WHITE,
        }
    }
    ode = Tex(
        "{\\ddot\\theta}({t})", "=",
        "-{\\mu} {\\dot\\theta}({t})",
        "-{g \\over L} \\sin\\big({\\theta}({t})\\big)",
        **tex_config,
    )
    return ode


def get_period_formula():
    return Tex(
        "2\\pi", "\\sqrt{\\,", "L", "/", "g", "}",
        tex_to_color_map={
            "L": BLUE,
            "g": YELLOW,
        }
    )


def pendulum_vector_field_func(point, mu=0.1, g=9.8, L=3):
    theta, omega = point[:2]
    return np.array([
        omega,
        -np.sqrt(g / L) * np.sin(theta) - mu * omega,
        0,
    ])


def get_vector_symbol(*texs, **kwargs):
    config = {
        "include_background_rectangle": True,
        "bracket_h_buff": SMALL_BUFF,
        "bracket_v_buff": SMALL_BUFF,
        "element_alignment_corner": ORIGIN,
    }
    config.update(kwargs)
    array = [[tex] for tex in texs]
    return Matrix(array, **config)


class Pendulum(VGroup):
    CONFIG = {
        "length": 3,
        "gravity": 9.8,
        "weight_diameter": 0.5,
        "initial_theta": 0.3,
        "omega": 0,
        "damping": 0.1,
        "top_point": 2 * UP,
        "rod_style": {
            "stroke_width": 3,
            "stroke_color": GREY_B,
            "sheen_direction": UP,
            "sheen_factor": 1,
        },
        "weight_style": {
            "stroke_width": 0,
            "fill_opacity": 1,
            "fill_color": GREY_BROWN, 
        },
        "dashed_line_config": {
            "num_dashes": 25,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "angle_arc_config": {
            "radius": 1,
            "stroke_color": WHITE,
            "stroke_width": 2,
        },
        "velocity_vector_config": {
            "color": RED,
        },
        "theta_label_height": 0.25,
        "set_theta_label_height_cap": False,
        "n_steps_per_frame": 100,
        "include_theta_label": True,
        "include_velocity_vector": False,
        "velocity_vector_multiple": 0.5,
        "max_velocity_vector_length_to_length_ratio": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_fixed_point()
        self.create_rod()
        self.create_weight()
        self.rotating_group = VGroup(self.rod, self.weight)
        self.create_dashed_line()
        self.create_angle_arc()
        if self.include_theta_label:
            self.add_theta_label()
        if self.include_velocity_vector:
            self.add_velocity_vector()

        self.set_theta(self.initial_theta)
        self.update()

    def create_fixed_point(self):
        self.fixed_point_tracker = VectorizedPoint(self.top_point)
        self.add(self.fixed_point_tracker)
        return self

    def create_rod(self):
        rod = self.rod = Line(UP, DOWN)
        rod.set_height(self.length)
        rod.move_to(self.get_fixed_point(), UP)
        self.add(rod)

    def create_weight(self):
        weight = self.weight = Circle()
        weight.set_width(self.weight_diameter)
        weight.set_style(**self.weight_style)
        weight.move_to(self.rod.get_end())
        self.add(weight)

    def create_dashed_line(self):
        line = self.dashed_line = DashedLine(
            self.get_fixed_point(),
            self.get_fixed_point() + self.length * DOWN,
            **self.dashed_line_config
        )
        line.add_updater(
            lambda l: l.move_to(self.get_fixed_point(), UP)
        )
        self.add_to_back(line)

    def create_angle_arc(self):
        self.angle_arc = always_redraw(lambda: Arc(
            arc_center=self.get_fixed_point(),
            start_angle=-90 * DEGREES,
            angle=self.get_arc_angle_theta(),
            **self.angle_arc_config,
        ))
        self.add(self.angle_arc)

    def get_arc_angle_theta(self):
        return self.get_theta()

    def add_velocity_vector(self):
        def make_vector():
            omega = self.get_omega()
            theta = self.get_theta()
            mvlr = self.max_velocity_vector_length_to_length_ratio
            max_len = mvlr * self.rod.get_length()
            vvm = self.velocity_vector_multiple
            multiple = np.clip(
                vvm * omega, -max_len, max_len
            )
            vector = Vector(
                multiple * RIGHT,
                **self.velocity_vector_config,
            )
            vector.rotate(theta, about_point=ORIGIN)
            vector.shift(self.rod.get_end())
            return vector

        self.velocity_vector = always_redraw(make_vector)
        self.add(self.velocity_vector)
        return self

    def add_theta_label(self):
        self.theta_label = always_redraw(self.get_label)
        self.add(self.theta_label)

    def get_label(self):
        label = Tex("\\theta")
        label.set_height(self.theta_label_height)
        if self.set_theta_label_height_cap:
            max_height = self.angle_arc.get_width()
            if label.get_height() > max_height:
                label.set_height(max_height)
        top = self.get_fixed_point()
        arc_center = self.angle_arc.point_from_proportion(0.5)
        vect = arc_center - top
        norm = get_norm(vect)
        vect = normalize(vect) * (norm + self.theta_label_height)
        label.move_to(top + vect)
        return label

    #
    def get_theta(self):
        theta = self.rod.get_angle() - self.dashed_line.get_angle()
        theta = (theta + PI) % TAU - PI
        return theta

    def set_theta(self, theta):
        self.rotating_group.rotate(
            theta - self.get_theta()
        )
        self.rotating_group.shift(
            self.get_fixed_point() - self.rod.get_start(),
        )
        return self

    def get_omega(self):
        return self.omega

    def set_omega(self, omega):
        self.omega = omega
        return self

    def get_fixed_point(self):
        return self.fixed_point_tracker.get_location()

    #
    def start_swinging(self):
        self.add_updater(Pendulum.update_by_gravity)

    def end_swinging(self):
        self.remove_updater(Pendulum.update_by_gravity)

    def update_by_gravity(self, dt):
        theta = self.get_theta()
        omega = self.get_omega()
        nspf = self.n_steps_per_frame
        for x in range(nspf):
            d_theta = omega * dt / nspf
            d_omega = op.add(
                -self.damping * omega,
                -(self.gravity / self.length) * np.sin(theta),
            ) * dt / nspf
            theta += d_theta
            omega += d_omega
        self.set_theta(theta)
        self.set_omega(omega)
        return self


class GravityVector(Vector):
    CONFIG = {
        "color": YELLOW,
        "length_multiple": 1 / 9.8,
    }

    def __init__(self, pendulum, **kwargs):
        super().__init__(DOWN, **kwargs)
        self.pendulum = pendulum
        self.scale(self.length_multiple * pendulum.gravity)
        self.attach_to_pendulum(pendulum)

    def attach_to_pendulum(self, pendulum):
        self.add_updater(lambda m: m.shift(
            pendulum.weight.get_center() - self.get_start(),
        ))

    def add_component_lines(self):
        self.component_lines = always_redraw(self.create_component_lines)
        self.add(self.component_lines)

    def create_component_lines(self):
        theta = self.pendulum.get_theta()
        x_new = rotate(RIGHT, theta)
        base = self.get_start()
        tip = self.get_end()
        vect = tip - base
        corner = base + x_new * np.dot(vect, x_new)
        kw = {"dash_length": 0.025}
        return VGroup(
            DashedLine(base, corner, **kw),
            DashedLine(corner, tip, **kw),
        )
    

class ThetaValueDisplay(VGroup):
    CONFIG = {

    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ThetaVsTAxes(Axes):
    CONFIG = {
        "x_range": [0,8],
        "y_range":[-PI/2,PI/2],
        "axis_config": {
            "color": "#EEEEEE",
            "stroke_width": 2,
            "include_tip": False,
        },
        "graph_style": {
            "stroke_color": GREEN,
            "stroke_width": 4,
            "fill_opacity": 0,
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_labels()

    def add_axes(self):
        self.axes = Axes(**self.axes_config)
        self.add(self.axes)

    def add_labels(self):
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        t_label = self.t_label = Tex("t")
        t_label.next_to(x_axis.get_right(), UP, MED_SMALL_BUFF)
        x_axis.label = t_label
        x_axis.add(t_label)
        theta_label = self.theta_label = Tex("\\theta(t)")
        theta_label.next_to(y_axis.get_top(), UP, SMALL_BUFF)
        y_axis.label = theta_label
        y_axis.add(theta_label)

        self.y_axis_label = theta_label
        self.x_axis_label = t_label

        x_axis.add_numbers()
        y_axis.add(self.get_y_axis_coordinates(y_axis))

    def get_y_axis_coordinates(self, y_axis):
        texs = [
            # "\\pi \\over 4",
            # "\\pi \\over 2",
            # "3 \\pi \\over 4",
            # "\\pi",
            "\\pi / 4",
            "\\pi / 2",
            "3 \\pi / 4",
            "\\pi",
        ]
        values = np.arange(1, 5) * PI / 4
        labels = VGroup()
        for pos_tex, pos_value in zip(texs, values):
            neg_tex = "-" + pos_tex
            neg_value = -1 * pos_value
            for tex, value in (pos_tex, pos_value), (neg_tex, neg_value):
                if value > self.y_range[1] or value < self.y_range[0]:
                    continue
                symbol = Tex(tex)
                symbol.scale(0.5)
                point = y_axis.number_to_point(value)
                symbol.next_to(point, LEFT, MED_SMALL_BUFF)
                labels.add(symbol)
        return labels

    def get_live_drawn_graph(self, pendulum,
                             t_max=None,
                             t_step=1.0 / 60,
                             **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_range[1]

        graph = VMobject()
        graph.set_style(**style)

        graph.all_coords = [(0, pendulum.get_theta())]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, pendulum.get_theta())
            if graph.time - graph.time_of_last_addition >= t_step:
                graph.all_coords.append(new_coords)
                graph.time_of_last_addition = graph.time
            points = [
                self.coords_to_point(*coords)
                for coords in [*graph.all_coords, new_coords]
            ]
            graph.set_points_smoothly(points)

        graph.add_updater(update_graph)
        return graph


class Vortrag(Scene):
    CONFIG = {
        "pendulum_config": {
            "initial_theta": 0.5,
            "length": 3,
            "top_point": 4 * RIGHT,
            "weight_diameter": 0.25,
            "gravity": 20,
        },
        "theta_vs_t_axes_config": {
            "height": 4,
            "y_range":[-1, 1],
            "x_range": [0,12,1],
            "axis_config": {
                "stroke_width": 2,
            }
        },
        "coordinate_plane_config": {
            "y_line_frequency": PI / 2,
            "x_line_frequency": 1,
            "y_axis_config": {
                "unit_size": 1,
            },
            "y_max": 4,
            "faded_line_ratio": 4,
            "background_line_style": {
                "stroke_width": 1,
            },
        },
        "little_pendulum_config": {
            "length": 1,
            "gravity": 4.9,
            "weight_diameter": 0.3,
            "include_theta_label": False,
            "include_velocity_vector": True,
            "angle_arc_config": {
                "radius": 0.2,
            },
            "velocity_vector_config": {
                "max_tip_length_to_length_ratio": 0.35,
                "max_stroke_width_to_length_ratio": 6,
            },
            "velocity_vector_multiple": 0.25,
            "max_velocity_vector_length_to_length_ratio": 0.8,
        },
        "big_pendulum_config": {
            "length": 1.6,
            "gravity": 4.9,
            "damping": 0.2,
            "weight_diameter": 0.3,
            "include_velocity_vector": True,
            "angle_arc_config": {
                "radius": 0.5,
            },
            "initial_theta": 80 * DEGREES,
            "omega": -1,
            "set_theta_label_height_cap": True,
        },
        "n_thetas": 11,
        "n_omegas": 7,
        "initial_grid_wait_time": 15,
        "vector_field_config": {
            "max_magnitude": 3,
        },
    }
    def construct(self):
    ## Titelseite
        bg = NumberPlane(
            axis_config = {
                "stroke_color": BLUE_D
            } 
        )
        field = VectorField(
            lambda x,y: np.array([x+y,x-y]),
            bg,
            magnitude_range = (0,8)
        )
        lines = StreamLines(
            lambda x,y: np.array([x+y,x-y]),
            bg,
            magnitude_range = (0,8)
        )
        anim_lines = AnimatedStreamLines(
            StreamLines(
                lambda x,y: np.array([x+y,x-y]),
                bg,
                magnitude_range = (0,5)
            )
        )
        title = Text(
            "Tutorium Analysis II", 
            font_size=72
        ).set_stroke(BLACK, 5, background=True)
        subtitle = Text(
            "- SoSe 2023 -", 
            font_size=60
        ).set_stroke(BLACK, 3, background=True)

        title1 = VGroup(
            title, 
            subtitle
        ).arrange(DOWN, buff = MED_SMALL_BUFF)
        rect = BackgroundRectangle(title1).scale(1.2)

        list = BulletedList(
            *[
                "Termine: Montag, Mittwoch, Freitag", 
                "Buch: Immer via Webseite aufrufen",
                "Zum Mitmachen: itempool.com/tutorium/live"
            ]
        )
        title2 = Text(
            "Tutorium Analysis II", 
            font_size=72
        ).set_stroke(BLACK, 5, background=True)
        title2.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title2, buff=0.05)
        underline.scale(1.25)


        self.add(bg, field, lines, anim_lines)
        self.add(rect, title1)
        self.interact()
        self.remove(anim_lines)
        self.interact()
        ## Ende

        
        ## Teil 1
        self.remove(
            field,
            lines,
            bg
        )
        self.play(
            FadeOut(subtitle),
            FadeOut(rect),
            title.animate.to_edge(UP, buff=MED_SMALL_BUFF),
            FadeIn(underline)
        )
        self.interact()
        list.fade_all_but(0, opacity=0.2)
        self.play(Write(list))
        self.interact()
        self.play(
            list.animate.fade_all_but(1, opacity=0.2)
        )
        self.interact()
        self.play(
            list.animate.fade_all_but(2, opacity=0.2)
        )
        self.interact()
        text = Text(
            "Wozu Analysis II?", 
            font_size=72
        )
        text.set_stroke(BLACK, 5, background=True)
        self.play(
            LaggedStart(
                FadeOut(Group(*self.mobjects)),
                Write(text),
                lag_ratio = 2*DEFAULT_LAGGED_START_LAG_RATIO
            )
        )
        pend = Pendulum(
            initial_theta = 0.6, 
            gravity = 9.8, 
            damping=0
        ).scale(1.25)
        self.interact()
        self.play(
            FadeOut(text),
            ShowCreation(pend),
            run_time = 2
        )
        self.interact()
        pend.start_swinging()
        self.interact()
        pend1 = Pendulum(initial_theta = 0.7).to_edge(LEFT).shift(0.5*RIGHT)
        L = Tex(
            "L"
        ).next_to(pend1.rod.get_center(), 2*RIGHT)
        m = Tex(
            "m"
        ).next_to(pend1.weight.get_center(), 2*RIGHT)
        pend.end_swinging()
        self.play(
            LaggedStart(
                FadeOut(pend),
                ShowCreation(pend1),
                Write(L),
                Write(m),
                lag_ratio = 2*DEFAULT_LAGGED_START_LAG_RATIO
            )
        )
        grav_vec = Arrow(
            pend1.weight.get_center(), 
            pend1.weight.get_center()-1.5*UP
        ).set_color(YELLOW)
        grav_vec_label = Tex(
            "\\vec{F}_G"
        ).set_color(YELLOW).next_to(grav_vec, LEFT)
        #self.interact()
        massenpunkt1 = VGroup(
            Text("Ruhelage:"), 
            Tex("(x(0),y(0))=(0,-L)")
        ).arrange(RIGHT, buff=MED_SMALL_BUFF)
        massenpunkt1.to_corner(UR, buff=LARGE_BUFF)
        massenpunkt2 = Tex(
            "(x(\\theta),y(\\theta))= ?"
        ).next_to(massenpunkt1, DOWN, buff=MED_SMALL_BUFF)
        massenpunkt3 = Tex(
            "(x(\\theta),y(\\theta))=(L\\sin(\\theta),-L\\cos(\\theta))"
        ).next_to(massenpunkt1, DOWN, buff=MED_SMALL_BUFF)
        mgFma = Tex(
            "m(0,-g)=\\vec F_G = m \\vec a"
        ).next_to(massenpunkt3, DOWN, buff=LARGE_BUFF)
        mgFma2 = Tex(
            "m(0,-g)=m(\\ddot x, \\ddot y)"
        ).next_to(massenpunkt3, DOWN, buff=LARGE_BUFF)
        mgFma3 = Tex(
            "(0,-g)=(\\ddot x, \\ddot y)"
        ).next_to(massenpunkt3, DOWN, buff=LARGE_BUFF)
        eq1 = Tex(
            "\\ddot \\theta \\cos(\\theta)=(\\dot \\theta)^2 \\sin(\\theta)"
        )
        eq2 = Tex(
            "-g=L\\ddot \\theta \\sin(\\theta)+L(\\dot \\theta)^2 \\cos(\\theta)"
        )
        eq = VGroup(eq1, eq2).arrange(DOWN, buff=MED_SMALL_BUFF)
        eq3 = Tex(
            "-\\frac{g}{L}=\\ddot \\theta \\sin(\\theta)+(\\dot \\theta)^2 \\cos(\\theta)"
        ).next_to(eq[0], DOWN, buff=MED_SMALL_BUFF)
        eq4 = Tex(
            "-\\frac{g}{L}\\sin(\\theta)=\\ddot \\theta \\sin^2(\\theta)+(\\dot \\theta)^2 \\sin(\\theta)\\cos(\\theta)"
        ).next_to(eq[0], DOWN, buff=MED_SMALL_BUFF)
        eq5 = Tex(
            "\\ddot \\theta=-\\frac{g}{L}\\sin(\\theta)"
        )
        eq6 = Tex(
            "\\ddot \\theta(t)=-\\frac{g}{L}\\sin(\\theta(t))"
        ).to_edge(RIGHT, buff=3.8*LARGE_BUFF)
        undjetzt = Text(
            "Und jetzt?"
        ).to_edge(RIGHT, buff=5*LARGE_BUFF)
        eq7 = Tex(
            "|\\theta(t)| \\ll 1 \\Rightarrow \\ddot \\theta(t) \\approx-\\frac{g}{L}\\theta(t)"
        ).to_edge(RIGHT, buff=2.5*LARGE_BUFF)
        eq8 = Tex(
            "\\ddot \\theta(t) =-\\frac{g}{L}\\theta(t)"
        ).to_edge(RIGHT, buff=4*LARGE_BUFF).shift(1.5*UP)
        eq9 = Tex(
            "\\theta(t)=\\theta_0 \\cos(\\sqrt{g/L}t)"
        ).to_edge(RIGHT, buff=3*LARGE_BUFF)
        #self.interact()
        self.play(Write(massenpunkt1))
        self.interact()
        self.play(Write(massenpunkt2))
        self.interact()
        self.play(
            Transform(massenpunkt2, massenpunkt3)
        )
        self.interact()
        self.play(
            ShowCreation(
                grav_vec
            ),
            ShowCreation(
                grav_vec_label
            )
        )
        self.interact()
        self.play(
            FadeOut(massenpunkt1),
            TransformMatchingTex(massenpunkt2, mgFma)
        )
        self.interact()
        self.play(
            Transform(mgFma, mgFma2)
        )
        self.interact()
        self.play(
            Transform(mgFma, mgFma3)
        )
        self.interact()
        self.play(
            FadeOut(
                Group(
                    mgFma, L, m,
                    pend1, grav_vec, grav_vec_label
                )
            ),
            Write(eq)
        )
        self.interact()
        self.play(
            Transform(
                eq[1], eq3
            )
        )
        self.interact()
        self.play(
            Transform(
                eq[1], eq4
            )
        )
        self.interact()
        self.play(
            FadeOut(eq),
            Write(eq5)
        )
        self.interact()
        self.play(
            FadeIn(
                Group(
                    L, m, pend1, 
                    grav_vec, grav_vec_label
                )
            ),
            eq5.animate.to_edge(RIGHT, buff=4*LARGE_BUFF)
        )
        self.interact()
        self.play(
            Transform(
                eq5, eq6
            )
        )
        self.interact()
        self.play(
            eq5.animate.shift(1.5*UP),
            Write(undjetzt)
        )
        self.interact()
        self.play(
            FadeOut(undjetzt),
            Write(eq7)
        )
        self.interact()
        self.play(
            FadeOut(eq5),
            eq7.animate.shift(1.5*UP)
        )
        self.wait()
        self.play(
            Transform(eq7, eq8)
        )
        self.interact()
        self.play(
            Write(eq9)
        )
        self.interact()
    ## Ende
        
        self.play(FadeOut(Group(*self.mobjects)))

    ## Teil 2  
        self.interact()
        self.add_pendulum()
        self.label_pendulum()
        self.add_graph()
        self.label_function()
        self.show_graph_period()
        self.show_length_and_gravity()
        self.interact()
        self.play(FadeOut(Group(*self.mobjects)))
    ## Ende


    ## Teil 3
        self.initialize_plane()
        self.interact()
        self.play(ShowCreation(self.plane))
        self.interact()
        self.show_state_with_pair_of_numbers()
        self.interact()
        self.show_evolution_from_a_start_state()
    # Ende


    ## Teil 4
        field2 = VectorField(
            self.vector_field_func,
            self.plane
        )
        self.interact()
        self.remove(
            self.trajectory,
            self.state,
            self.state_dot,
            self.h_line,
            self.v_line
        )
        self.interact()
        self.show_acceleration_dependence()
        self.interact()
        self.play(
            Write(field2)
        )
        self.interact()
        self.add_flexible_state()
        st = self.state
        dot = self.get_state_dot(st)
        self.search_set = VGroup(dot)
        self.tie_state_to_dot_position(st, dot)
        self.add(dot)
        self.interact()
        self.play(
            dot.animate.move_to(self.plane.c2p(2,0))
        )
        self.interact()
        self.play(
            dot.animate.move_to(self.plane.c2p(1/2,0))
        )
        self.interact()
        trace = self.get_evolving_trajectory(dot).set_stroke(BLUE_E, 4)
        path = self.get_path(
            [dot.get_center()[0],dot.get_center()[1]]
        ).set_opacity(0)
        self.interact()
        self.add(trace)
        self.play(
            ShowCreation(
                path,
                rate_func=linear,
            ),
            UpdateFromFunc(
                dot, lambda d: d.move_to(path.get_points()[-1])
            ),
            run_time = 20
        )
        self.interact()
        self.remove(trace, path)
        self.interact()
        self.interact()
        self.interact()
    ## Ende
    
    ## Interaktion
    def on_mouse_press(self, point, button, mods):
        super().on_mouse_press(point, button, mods)
        mob = self.point_to_mobject(point, search_set=self.search_set)
        if mob is None:
            return
        self.mouse_drag_point.move_to(point) 
        mob.add_updater(
            lambda x: x.move_to(
                self.mouse_drag_point
            ) 
        )
        self.unlock_mobject_data()
        self.lock_static_mobject_data()

    def on_mouse_release(self, point, button, mods):
        super().on_mouse_release(point, button, mods)
        self.search_set.clear_updaters() 
    ##


    #################################################
    #                 Definitionen                  #
    ################################################# 

    def get_path(self, pos):

        def DGL(t, point):
            x_dot = self.vector_field_func(point[0],point[1])[0]
            y_dot = self.vector_field_func(point[0],point[1])[1]
            return [x_dot, y_dot]
        
        def path(init): 
            grid = np.arange(0.0, 60.0, 0.01)
            res = odeint(DGL, init, grid, tfirst=True)
            p = VMobject()
            p.set_points_as_corners([*[[a,b,0] for a,b in zip(res[:,0],res[:,1])]])
            p.set_stroke(None,1)
            p.make_smooth() 
            return p
        
        return path(pos)
        

    def add_pendulum(self):
        pendulum = self.pendulum = Pendulum(**self.pendulum_config)
        pendulum.start_swinging()
        frame = self.camera.frame
        frame.save_state()
        frame.scale(0.5)
        frame.move_to(pendulum.dashed_line)

        self.add(pendulum, frame)


    def label_pendulum(self):
        pendulum = self.pendulum
        label = pendulum.theta_label
        rect = SurroundingRectangle(label, buff=0.5 * SMALL_BUFF)
        rect.add_updater(lambda r: r.move_to(label))

        self.interact()
        self.play(
            ShowCreationThenFadeOut(rect),
        )


    def add_graph(self):
        axes = self.axes = ThetaVsTAxes(**self.theta_vs_t_axes_config)
        axes.y_axis.label.next_to(axes.y_axis, UP, buff=0)
        axes.to_corner(UL)

        self.interact()
        self.play(
            Restore(
                self.camera.frame,
                rate_func=squish_rate_func(smooth, 0, 0.9),
            ),
            DrawBorderThenFill(
                axes,
                rate_func=squish_rate_func(smooth, 0.5, 1),
                lag_ratio=0.9,
            ),
            Transform(
                self.pendulum.theta_label.copy().clear_updaters(),
                axes.y_axis.label.copy(),
                remover=True,
                rate_func=squish_rate_func(smooth, 0, 0.8),
            ),
            run_time=3,
        )
        self.interact()
        self.graph = axes.get_live_drawn_graph(self.pendulum)
        self.add(self.graph)


    def label_function(self):
        hm_word = TexText("Harmonische Schwingung")
        hm_word.scale(1.25)
        hm_word.to_edge(UP)

        formula = Tex(
            "=\\theta_0 \\cos(\\sqrt{g / L} t)"
        )
        formula.next_to(
            self.axes.y_axis_label, RIGHT, SMALL_BUFF
        )
        formula.set_stroke(width=0, background=True)
        self.interact()
        self.play(FadeIn(hm_word, DOWN))
        self.interact()
        self.play(
            Write(formula),
            hm_word.to_corner, UR
        )


    def show_graph_period(self):
        pendulum = self.pendulum
        axes = self.axes

        period = self.period = TAU * np.sqrt(
            pendulum.length / pendulum.gravity
        )
        amplitude = pendulum.initial_theta

        line = Line(
            axes.coords_to_point(0, amplitude),
            axes.coords_to_point(period, amplitude),
        )
        line.shift(SMALL_BUFF * RIGHT)
        brace = Brace(line, UP, buff=SMALL_BUFF)
        brace.add_to_back(brace.copy().set_style(BLACK, 10))
        formula = get_period_formula()
        formula.next_to(brace, UP, SMALL_BUFF)

        self.period_formula = formula
        self.period_brace = brace
        self.interact()
        self.play(
            GrowFromCenter(brace),
            FadeIn(formula),
        )


    def show_length_and_gravity(self):
        formula = self.period_formula
        L = formula.get_part_by_tex("L") 
        g = formula.get_part_by_tex("g")
        
        rod = self.pendulum.rod
        new_rod = rod.copy()
        new_rod.set_stroke(BLUE, 7)
        new_rod.add_updater(lambda r: r.put_start_and_end_on(
            *rod.get_start_and_end()
        ))

        g_vect = GravityVector(
            self.pendulum,
            length_multiple=0.5 / 9.8,
        )
        self.interact()
        self.play(
            ShowCreationThenDestructionAround(L),
            ShowCreation(new_rod),
        )
        self.interact()
        self.play(FadeOut(new_rod))
        self.interact()
        self.play(
            ShowCreationThenDestructionAround(g),
            GrowArrow(g_vect),
        )
        self.gravity_vector = g_vect


    def tweak_length_and_gravity(self):
        pendulum = self.pendulum
        axes = self.axes
        graph = self.graph
        brace = self.period_brace
        formula = self.period_formula
        g_vect = self.gravity_vector

        graph.clear_updaters()
        period2 = self.period * np.sqrt(2)
        period3 = self.period / np.sqrt(2)
        amplitude = pendulum.initial_theta
        graph2, graph3 = [
            axes.get_graph(
                lambda t: amplitude * np.cos(TAU * t / p),
                color=RED,
            )
            for p in (period2, period3)
        ]
        formula.add_updater(lambda m: m.next_to(
            brace, UP, SMALL_BUFF
        ))

        new_pendulum_config = dict(self.pendulum_config)
        new_pendulum_config["length"] *= 2
        new_pendulum_config["top_point"] += 3.5 * UP
        new_pendulum = Pendulum(**new_pendulum_config)

    
        g_vect.attach_to_pendulum(new_pendulum)
        self.interact()
        new_pendulum.start_swinging()
        self.interact()
        self.play(
            ReplacementTransform(graph, graph2),
            brace.stretch, np.sqrt(2), 0, {"about_edge": LEFT},
        )
        self.add(g_vect)
        self.interact()

        new_pendulum.gravity *= 4
        g_vect.scale(2)
        self.play(
            FadeIn(graph3),
            brace.stretch, 0.5, 0, {"about_edge": LEFT},
        )
        self.interact()
    

    def initialize_plane(self):
        plane = self.plane = NumberPlane(
            **self.coordinate_plane_config
        )
        plane.axis_labels = VGroup(
            plane.get_x_axis_label(
                "\\theta", RIGHT, UL, buff=SMALL_BUFF
            ),
            plane.get_y_axis_label(
                "\\dot \\theta", UP, DR, buff=SMALL_BUFF
            ).set_color(YELLOW),
        )
        for label in plane.axis_labels:
            label.add_background_rectangle()
        plane.add(plane.axis_labels)

        plane.y_axis.add_numbers(direction=DL)

        x_axis = plane.x_axis
        label_texs = ["\\pi \\over 2", "\\pi", "3\\pi \\over 2", "\\tau"]
        values = [PI / 2, PI, 3 * PI / 2, TAU]
        x_axis.coordinate_labels = VGroup()
        x_axis.add(x_axis.coordinate_labels)
        for value, label_tex in zip(values, label_texs):
            for u in [-1, 1]:
                tex = label_tex
                if u < 0:
                    tex = "-" + tex
                label = Tex(tex)
                label.scale(0.5)
                if label.get_height() > 0.4:
                    label.set_height(0.4)
                point = x_axis.number_to_point(u * value)
                label.next_to(point, DR, SMALL_BUFF)
                x_axis.coordinate_labels.add(label)
        return plane


    def show_state_with_pair_of_numbers(self):
        axis_labels = self.plane.axis_labels

        state = self.get_flexible_state_picture()
        dot = self.get_state_controlling_dot(state)
        h_line = always_redraw(
            lambda: self.get_tracking_h_line(dot.get_center())
        )
        v_line = always_redraw(
            lambda: self.get_tracking_v_line(dot.get_center())
        )

        self.add(dot)
        anims = [GrowFromPoint(state, dot.get_center())]
        if hasattr(self, "state_dots"):
            anims.append(FadeOut(self.state_dots))
        self.play(*anims)
        self.interact()
        for line, label in zip([h_line, v_line], axis_labels):
            # self.add(line, dot)
            self.play(
                ShowCreation(line),
                ShowCreationThenFadeAround(label),
                run_time=1
            )
            self.interact()
        for vect in LEFT, 3 * UP:
            self.play(
                ApplyMethod(
                    dot.shift, vect,
                    rate_func=there_and_back,
                    run_time=2,
                )
            )
            self.interact()
        self.wait()
        for vect in 2 * LEFT, 3 * UP, 2 * RIGHT, 2 * DOWN:
            self.play(dot.shift, vect, run_time=1.5)
            self.interact()
        self.wait()

        self.state = state
        self.state_dot = dot
        self.h_line = h_line
        self.v_line = v_line


    def show_acceleration_dependence(self):
        ode = get_ode()
        thetas = ode.get_parts_by_tex("\\theta")
        thetas[0].set_color(RED)
        thetas[1].set_color(YELLOW)
        ode.move_to(
            FRAME_WIDTH * RIGHT / 4 +
            FRAME_HEIGHT * UP / 4,
        )
        ode.add_background_rectangle_to_submobjects()

        self.play(Write(ode))
        self.interact()
        self.play(FadeOut(ode))


    def show_evolution_from_a_start_state(self):
        state = self.state
        dot = self.state_dot

        self.play(
            Rotating(
                dot,
                about_point=dot.get_center() + UR,
                rate_func=smooth,
            )
        )
        self.interact()

        # Show initial trajectory
        state.pendulum.clear_updaters(recurse=False)
        self.tie_dot_position_to_state(dot, state)
        state.pendulum.start_swinging()
        trajectory = self.trajectory = self.get_evolving_trajectory(dot)
        self.add(trajectory)
        
    
    def get_down_vectors(self):
        down_vectors = VGroup(*[
            Vector(0.5 * DOWN)
            for x in range(10 * 150)
        ])
        down_vectors.arrange_in_grid(10, 150, buff=MED_SMALL_BUFF)
        down_vectors.set_color_by_gradient(BLUE, RED)
        down_vectors.to_edge(RIGHT)
        return down_vectors


    def get_down_vectors_animation(self, down_vectors):
        return LaggedStart(
            *[
                GrowArrow(v, rate_func=there_and_back)
                for v in down_vectors
            ],
            lag_ratio=0.0005,
            run_time=2,
            remover=True
        )


    def get_initial_thetas(self):
        angle = 3 * PI / 4
        return np.linspace(-angle, angle, self.n_thetas)


    def get_initial_omegas(self):
        return np.linspace(-1.5, 1.5, self.n_omegas)


    def get_state(self, pendulum):
        return (pendulum.get_theta(), pendulum.get_omega())


    def get_state_point(self, pendulum):
        return self.plane.coords_to_point(
            *self.get_state(pendulum)
        )


    def get_flexible_state_picture(self):
        buff = MED_SMALL_BUFF
        height = FRAME_HEIGHT / 2 - buff
        rect = Square(
            side_length=height,
            stroke_color=WHITE,
            stroke_width=2,
            fill_color="#111111",
            fill_opacity=1,
        )
        rect.to_corner(UL, buff=buff / 2)
        pendulum = Pendulum(
            top_point=rect.get_center(),
            **self.big_pendulum_config
        )
        pendulum.fixed_point_tracker.add_updater(
            lambda m: m.move_to(rect.get_center())
        )

        state = VGroup(rect, pendulum)
        state.rect = rect
        state.pendulum = pendulum
        return state


    def get_state_dot(self, state):
        dot = Dot(color=PINK)
        dot.move_to(self.get_state_point(state.pendulum))
        return dot


    def get_state_controlling_dot(self, state):
        dot = self.get_state_dot(state)
        self.tie_state_to_dot_position(state, dot)
        return dot


    def tie_state_to_dot_position(self, state, dot):
        def update_pendulum(pend):
            theta, omega = self.plane.point_to_coords(
                dot.get_center()
            )
            pend.set_theta(theta)
            pend.set_omega(omega)
            return pend
        state.pendulum.add_updater(update_pendulum)
        state.pendulum.get_arc_angle_theta = \
            lambda: self.plane.x_axis.point_to_number(dot.get_center())
        return self


    def tie_dot_position_to_state(self, dot, state):
        dot.add_updater(lambda d: d.move_to(
            self.get_state_point(state.pendulum)
        ))
        return dot


    def get_tracking_line(self, point, axis, color=WHITE):
        number = axis.point_to_number(point)
        axis_point = axis.number_to_point(number)
        return DashedLine(
            axis_point, point,
            dash_length=0.025,
            color=color,
        )


    def get_tracking_h_line(self, point):
        return self.get_tracking_line(
            point, self.plane.y_axis, WHITE,
        )


    def get_tracking_v_line(self, point):
        return self.get_tracking_line(
            point, self.plane.x_axis, YELLOW,
        )
        

    def get_evolving_trajectory(self, mobject):
        trajectory = TracedPath(
            lambda: mobject.get_center()
        ).set_stroke(RED, 3)
        return trajectory


    def add_flexible_state(self):
        self.state = self.get_flexible_state_picture()
        self.add(self.state)


    def show_trajectory(self):
        state = self.state
        dot = self.state_dot

        state.pendulum.clear_updaters(recurse=False)
        self.tie_dot_position_to_state(dot, state)
        state.pendulum.start_swinging()

        trajectory = self.get_evolving_trajectory(dot)
        trajectory.set_stroke(WHITE, 3)

        self.add(trajectory, dot)
        self.wait(25)


    def vector_field_func(self, x, y):
        mu, g, L = [
            self.big_pendulum_config.get(key)
            for key in ["damping", "gravity", "length"]
        ]
        return pendulum_vector_field_func(
            x * RIGHT + y * UP,
            mu=mu, g=g, L=L
        )
