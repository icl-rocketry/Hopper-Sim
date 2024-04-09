def play_animation(self, rocket_len, state_traj, control_traj, state_traj_ref=None, control_traj_ref=None,
                    save_option=0, dt=0.1,
                    title='Rocket Powered Landing'):
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Upward (m)')
    ax.set_zlim(0, 10)
    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)
    ax.set_title(title, pad=20, fontsize=15)


    # trajectory formulation
    position = self.get_rocket_body_position(rocket_len, state_traj, control_traj)
    sim_horizon = np.size(position, 0)
    for t in range(np.size(position, 0)):
        x = position[t, 0]
        if x < 0:
            sim_horizon = t
            break
        
    # animation
    line_traj, = ax.plot(position[:1, 1], position[:1, 2], position[:1, 0])
    xg, yg, zg, xh, yh, zh, xf, yf, zf = position[0, 3:]
    line_rocket, = ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')
    line_force, = ax.plot([yg, yf], [zg, zf], [xg, xf], linewidth=2, color='red')

    # reference data
    if state_traj_ref is None or control_traj_ref is None:
        position_ref=numpy.zeros_like(position)
        sim_horizon_ref=sim_horizon
    else:
        position_ref = self.get_rocket_body_position(rocket_len, state_traj_ref, control_traj_ref)
        sim_horizon_ref = np.size((position_ref,0))
        for t in range(np.size(position_ref, 0)):
            x = position_ref[t, 0]
            if x < 0:
                sim_horizon_ref = t
                break
    # animation
    line_traj_ref, = ax.plot(position_ref[:1, 1], position_ref[:1, 2], position_ref[:1, 0], linewidth=2, color='gray', alpha=0.5)
    xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[0, 3:]
    line_rocket_ref, = ax.plot([yg_ref, yh_ref], [zg_ref, zh_ref], [xg_ref, xh_ref], linewidth=5, color='gray', alpha=0.5)
    line_force_ref, = ax.plot([yg_ref, yf_ref], [zg_ref, zf_ref], [xg_ref, xf_ref], linewidth=2, color='red', alpha=0.2)

    # time label
    time_template = 'time = %.1fs'
    time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)
    # time_text = ax.text2D(0.66, 0.65, "time", transform=ax.transAxes)
    # time_text = ax.text2D(0.50, 0.65, "time", transform=ax.transAxes)

    # customize
    if state_traj_ref is not None or control_traj_ref is not None:
        plt.legend([line_traj, line_traj_ref], ['learned', 'truth'], ncol=1, loc='best',
                    bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

    def update_traj(num):
        # customize
        time_text.set_text(time_template % (num * dt))

        # trajectory
        if num> sim_horizon:
            t=sim_horizon
        else:
            t=num
        line_traj.set_data(position[:t, 1], position[:t, 2])
        line_traj.set_3d_properties(position[:t, 0])

        # rocket
        xg, yg, zg, xh, yh, zh, xf, yf, zf = position[t, 3:]
        line_rocket.set_data([yg, yh], [zg, zh])
        line_rocket.set_3d_properties([xg, xh])
        line_force.set_data([yg, yf], [zg, zf])
        line_force.set_3d_properties([xg, xf])

        # reference
        if num> sim_horizon_ref:
            t_ref=sim_horizon_ref
        else:
            t_ref=num
        line_traj_ref.set_data(position_ref[:t_ref, 1], position_ref[:t_ref, 2])
        line_traj_ref.set_3d_properties(position_ref[:t_ref, 0])

        # rocket
        xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[num, 3:]
        line_rocket_ref.set_data([yg_ref, yh_ref], [zg_ref, zh_ref])
        line_rocket_ref.set_3d_properties([xg_ref, xh_ref])
        line_force_ref.set_data([yg_ref, yf_ref], [zg_ref, zf_ref])
        line_force_ref.set_3d_properties([xg_ref, xf_ref])


        return line_traj, line_rocket, line_force, line_traj_ref, line_rocket_ref, line_force_ref,  time_text

    ani = animation.FuncAnimation(fig, update_traj, max(sim_horizon,sim_horizon_ref), interval=100, blit=True)

    if save_option != 0:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
        ani.save(title + '.mp4', writer=writer, dpi=300)
        print('save_success')

    plt.show()
    
    
def get_rocket_body_position(self, rocket_len, state_traj, control_traj):

    # thrust_position in body frame
    r_T_B = vertcat(-(rocket_len / 2), 0, 0)

    # horizon
    horizon = np.size(control_traj, 0)
    
    # for normalization in the plot
    norm_f = np.linalg.norm(control_traj, axis=1);
    max_f = np.amax(norm_f)
    position = np.zeros((horizon, 12))
    
    for t in range(horizon):    
        # position of COM
        rc = state_traj[t, 0:3]
        # altitude of quaternion
        q = state_traj[t, 6:10]
        # thrust force
        f = control_traj[t, 0:3]

        # direction cosine matrix from body to inertial
        CIB = np.transpose(self.dir_cosine(q).full())

        # position of gimbal point (rocket tail)
        rg = rc + mtimes(CIB, r_T_B).full().flatten()
        # position of rocket tip
        rh = rc - mtimes(CIB, r_T_B).full().flatten()

        # direction of force
        df = np.dot(CIB, f) / max_f
        rf = rg - df

        # store
        position[t, 0:3] = rc
        position[t, 3:6] = rg
        position[t, 6:9] = rh
        position[t, 9:12] = rf

    return position