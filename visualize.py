### plotting tools for 2x2 analysis
import plotly.graph_objs as go
import numpy as np
import h5py
import yaml

def get_spillIDs(light_wvfm, light_trig, packets, tracks):
    ### 
    ### code written by Angela White, from larnd-sim/examples
    ## DEFINE COMMON VALUES
    SPILL_PERIOD = 1.2e7
    SAMPLES = len(light_wvfm[0][0])
    BIT = min(x for x in abs(light_wvfm[0][0]) if x != 0)
    
    pack_type = np.array(packets['packet_type'])
    p_tstamp = np.array(packets['timestamp'])
    io_group = np.array(packets['io_group'])
    io_channel = np.array(packets['io_channel'])
    l_tsync = np.array(light_trig['ts_sync'])
    spillID = np.array(tracks['eventID'])
    trackID = np.array(tracks['trackID'])
    opt_chan = np.array(light_trig['op_channel'])
    tstamp_trig0 = p_tstamp[pack_type==0]
    tstamp_trig7 = p_tstamp[pack_type==7]

    ## IDENTIFY THE INDEX WHERE THE TURNOVER OCCURS
    charge_cutoff = np.where(tstamp_trig0 > 1.999**31)[0][-1]
    light_cutoff = np.where(tstamp_trig7 > 1.999**31)[0][-1]
    wvfm_cutoff = np.where(l_tsync > 1.999**31)[0][-1]

    ## ADD 2^31 TO ALL TIMESTAMPS FOLLOWING THE TURNOVER
    tstamp_real_trig0 = np.concatenate((tstamp_trig0[:(charge_cutoff+1)],((2**31)+tstamp_trig0[(charge_cutoff+1):])))
    tstamp_real_trig7 = np.concatenate((tstamp_trig7[:(light_cutoff+1)],((2**31)+tstamp_trig7[(light_cutoff+1):])))
    l_tsync_real = np.concatenate((l_tsync[:(wvfm_cutoff+1)],((2**31)+l_tsync[(wvfm_cutoff+1):])))

    ## DEFINE SPILLID (EVENTID) FOR PACKETS AND LIGHT
    light_spillIDs = (np.rint(l_tsync_real/SPILL_PERIOD)).astype(int)
    packet0_spillIDs = (np.rint(tstamp_real_trig0/SPILL_PERIOD)).astype(int)
    packet7_spillIDs = (np.rint(tstamp_real_trig7/SPILL_PERIOD)).astype(int)
    return light_spillIDs, packet0_spillIDs, packet7_spillIDs

def plot_spill_MC(light_trig, hits, light_spillIDs, eventID):
    hits_event = hits[hits['edep_event_ids'] == eventID]
    hits_event_x = np.array(hits_event['z'])
    hits_event_y = np.array(hits_event['y'])
    hits_event_z = np.array(hits_event['x'])
    hits_event_t = np.array(hits_event['t'])
    hits_event_q = np.array(hits_event['q'])

    # need to get t0 for each module to get drift coordinate
    light_trig_event = light_trig[light_spillIDs == eventID]
    light_spillIDs_event = light_trig_event['op_channel'][:,0]

    # for each module, find the light trig with the smallest timestamp -> use for t0 calculation.
    # modules without a light trigger are indicated by a -1 in place of timestamp
    ts_sync_min_event = [] # ns
    for min_op_channel in np.unique(light_trig['op_channel'][:,0]):
        module_ts_sync = light_trig_event[light_spillIDs_event == min_op_channel]['ts_sync']
        if module_ts_sync.size > 0:
            module_tlight = int(np.min(module_ts_sync)) * 0.1 * 1e3 
            ts_sync_min_event.append(module_tlight)
        else:
            ts_sync_min_event.append(-1)
    print(ts_sync_min_event)
    
    detprop_path = '/sdf/home/s/sfogarty/Desktop/RadDecay/neutron_TOF/analysis_tools_2x2/2x2.yaml'
    with open(detprop_path) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)
    tpc_to_op_channel = detprop['tpc_to_op_channel']

    # get relationship between module to op channel
    i,j = 0,0
    module_to_op_channel = np.zeros((int(len(tpc_to_op_channel)/2), len(tpc_to_op_channel[0]*2)))
    while i < len(tpc_to_op_channel):
        module_to_op_channel[j] = np.concatenate((tpc_to_op_channel[i], tpc_to_op_channel[i+1]))
        j += 1
        i += 2

    module_to_io_group =detprop['module_to_io_groups']
    io_group_to_module = {value: key for key, values in module_to_io_group.items() for value in values}

    # calculate drift coordinate
    v_drift = 1.6 / 1e3 # mm/ns
    for i,light_ts_sync in enumerate(ts_sync_min_event):
        IOs = module_to_io_group[i+1]
        for IO in IOs:
            hits_event_mask = (hits_event['io_group'] == IO)
            if IO % 2 == 1:
                hits_event_x[hits_event_mask] = hits_event_x[hits_event_mask] + v_drift * np.abs(hits_event_t[hits_event_mask] - light_ts_sync)
            else:
                hits_event_x[hits_event_mask] = hits_event_x[hits_event_mask] - v_drift * np.abs(hits_event_t[hits_event_mask] - light_ts_sync)

    fig = go.Figure(data=[go.Scatter3d(x=hits_event_x, y=hits_event_z,\
                                       z=hits_event_y, mode='markers',\
                    marker=dict(size=1,color=hits_event_q, colorscale='Viridis',\
                    colorbar=dict(title='charge [ke-]')))])

    # Define the coordinates of the planes
    x_plane = np.array([-655, 655])
    z_plane = np.array([-240, 1100])
    X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
    Y_plane = np.array([0, 0])
    opacity = 0.2
    fig.add_trace(go.Surface(x=X_plane, y=Y_plane, z=Z_plane, colorscale='Greys', opacity=opacity, showscale=False))
    fig.add_trace(go.Surface(x=Y_plane, y=X_plane, z=Z_plane, colorscale='Greys', opacity=opacity, showscale=False))

    fig.update_layout(scene=dict(xaxis=dict(range=[-655, 655],title='x (drift) [mm]', tickfont=dict(size=10)), \
                                 yaxis=dict(range=[-655, 655], title='z [mm]', tickfont=dict(size=10))\
                                 ,zaxis=dict(range=[-240, 1100], title='y [mm]', tickfont=dict(size=10)),\
                                 aspectratio=dict(x=2, y=2, z=2)))

    fig.show(config={'displayModeBar': False}, width=800, height=500)
