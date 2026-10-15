"""Seed-aware plots and actual-trajectory animations for the Taxi project."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, colors, patches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


_INK = "#18364B"
_MUTED = "#63788A"
_PAPER = "#FAFAF5"
_PALETTE = ["#187E83", "#E38645", "#6477B1", "#B85475", "#8D9550", "#735A91"]
_LANDMARK_NAMES = ("R", "G", "Y", "B")
_LANDMARK_COLORS = ("#CC5A59", "#278868", "#BE9123", "#3978AD")
_ACTION_NAMES = ("South", "North", "East", "West", "Pickup", "Dropoff")
_ACTION_SYMBOLS = ("↓", "↑", "→", "←", "P", "D")
_TAXI_MAP = ("+---------+", "|R: | : :G|", "| : | : : |", "| : : : : |",
             "| | : | : |", "|Y| : |B: |", "+---------+")


def _axis_style(ax):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E8EC", linewidth=.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#C2D0D8")
    ax.tick_params(colors=_MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)


def _map_lines(env):
    raw = getattr(env, "MAP", getattr(env, "desc", _TAXI_MAP))
    if isinstance(raw, str):
        return raw.strip("\n").splitlines()
    result = []
    for row in raw:
        if isinstance(row, bytes):
            result.append(row.decode("ascii"))
        elif isinstance(row, str):
            result.append(row)
        else:
            result.append("".join(item.decode("ascii") if isinstance(item, bytes)
                                  else str(item) for item in row))
    return result


def _road_open(env, first, second):
    r0, c0 = first
    r1, c1 = second
    if abs(r1-r0) + abs(c1-c0) != 1:
        return False
    if r0 != r1:
        return True
    return _map_lines(env)[r0+1][2*min(c0, c1)+2] != "|"


def _draw_car(ax, row, col, color="#E6A238"):
    # Plain vector patches keep the notebook portable and visually consistent.
    cabin = patches.FancyBboxPatch((col-.17, row-.24), .34, .21,
                                  boxstyle="round,pad=.015,rounding_size=.055",
                                  facecolor=color, edgecolor=_INK, linewidth=.8, zorder=22)
    body = patches.FancyBboxPatch((col-.28, row-.08), .56, .25,
                                 boxstyle="round,pad=.01,rounding_size=.05",
                                 facecolor=color, edgecolor=_INK, linewidth=1, zorder=23)
    ax.add_patch(cabin)
    ax.add_patch(body)
    ax.add_patch(patches.Rectangle((col-.10, row-.21), .20, .11,
                                   facecolor="#E4F3F7", edgecolor="none", zorder=24))
    for dx in (-.18, .18):
        ax.add_patch(patches.Circle((col+dx, row+.17), .061,
                                   facecolor=_INK, edgecolor="white", linewidth=.5, zorder=24))


def _board(ax, env, title="", passenger=0, destination=3,
           values=None, actions=None, taxi=None, path=None):
    rows, cols = int(env.rows), int(env.cols)
    landmarks = tuple(tuple(x) for x in env.landmarks)
    cmap = colors.LinearSegmentedColormap.from_list(
        "taxi_value", ["#31566A", "#89BEC2", "#F0F7EC"])
    norm = None
    if values is not None and np.any(np.isfinite(values)):
        finite = np.asarray(values)[np.isfinite(values)]
        low, high = float(np.min(finite)), float(np.max(finite))
        if np.isclose(low, high):
            low, high = low-1, high+1
        norm = colors.Normalize(low, high)
    for row in range(rows):
        for col in range(cols):
            face = "#F0F4F3"
            if norm is not None and np.isfinite(values[row, col]):
                face = cmap(norm(values[row, col]))
            elif values is not None:
                face = "#E6E9E7"
            ax.add_patch(patches.Rectangle((col-.5, row-.5), 1, 1,
                                           facecolor=face, edgecolor="white",
                                           linewidth=1.1, zorder=1))
            if actions is not None and actions[row, col] >= 0:
                ink = "white" if norm is not None and norm(values[row, col]) < .30 else _INK
                action = int(actions[row, col])
                ax.text(col, row-.07, _ACTION_SYMBOLS[action], ha="center", va="center",
                        color=ink, weight="bold", fontsize=18 if action < 4 else 12,
                        bbox=dict(boxstyle="circle,pad=.18", facecolor="#FFFFFF40", edgecolor="none")
                        if action >= 4 else None, zorder=5)
                ax.text(col, row+.27, f"{values[row,col]:.1f}", ha="center", va="center",
                        color=ink, fontsize=7.7, zorder=5)
    if path is not None:
        states = path["states"] if isinstance(path, dict) else path
        decoded = [tuple(env.decode(int(s))) for s in states]
        for first, second in zip(decoded[:-1], decoded[1:]):
            if _road_open(env, first[:2], second[:2]):
                color = _PALETTE[1] if first[2] < 4 else _PALETTE[0]
                ax.plot([first[1], second[1]], [first[0], second[0]],
                        color=color, alpha=.70, linewidth=3.4,
                        solid_capstyle="round", zorder=7)
    for index, ((row, col), name, color) in enumerate(zip(
            landmarks, _LANDMARK_NAMES, _LANDMARK_COLORS)):
        ax.text(col-.33, row-.30, name, ha="center", va="center", fontsize=8,
                weight="bold", color="white", zorder=15,
                bbox=dict(boxstyle="round,pad=.18", facecolor=color, edgecolor="none"))
        if index == int(destination):
            ax.plot([col+.30, col+.30], [row-.31, row+.03], color=_INK,
                    linewidth=1.4, zorder=16)
            ax.add_patch(patches.Polygon([[col+.30,row-.31], [col+.46,row-.23],
                                          [col+.30,row-.15]], facecolor="#CA5D54",
                                         edgecolor="none", zorder=17))
        if int(passenger) == index and int(passenger) != int(destination):
            ax.add_patch(patches.Rectangle((col-.39, row+.16), .16, .16,
                                           facecolor="#9D5079", edgecolor="white",
                                           linewidth=.8, zorder=17))
    text_map = _map_lines(env)
    for row in range(rows):
        for col in range(cols-1):
            if text_map[row+1][2*col+2] == "|":
                ax.plot([col+.5, col+.5], [row-.5, row+.5],
                        color=_INK, linewidth=3.2, zorder=20)
    ax.add_patch(patches.Rectangle((-.5, -.5), cols, rows, fill=False,
                                   edgecolor=_INK, linewidth=2, zorder=20))
    if taxi is not None:
        _draw_car(ax, *taxi, color="#E6A238" if int(passenger) < 4 else "#42A48B")
    ax.set_xlim(-.57, cols-.43)
    ax.set_ylim(rows-.43, -.57)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(length=0, colors=_MUTED, labelsize=8, pad=4)
    ax.set_xlabel("column", fontsize=8, color=_MUTED)
    ax.set_ylabel("row", fontsize=8, color=_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11.5, color=_INK, weight="bold", pad=12)
    return ax


def draw_taxi_map(ax, env, state, title="", path=None, show_policy=False, q=None):
    """Draw true road walls, passenger, destination, taxi and observed road moves.

    Path segments are drawn only between adjacent cells connected by a road.
    Orange means travelling toward pickup; teal means carrying the passenger.
    Policy arrows hold the supplied state's passenger/destination context fixed.
    """
    row, col, passenger, destination = map(int, env.decode(int(state)))
    values = actions = None
    if show_policy:
        if q is None:
            raise ValueError("q is required when show_policy=True")
        values, actions = _slice_values(env, q, passenger, destination)
    _board(ax, env, title=title, passenger=passenger, destination=destination,
           values=values, actions=actions, taxi=(row,col), path=path)
    passenger_text = "ON BOARD" if passenger == 4 else (
        "DELIVERED" if passenger == destination else f"waiting at {_LANDMARK_NAMES[passenger]}")
    ax.text(.5, -.14, f"Passenger: {passenger_text}  |  Destination: {_LANDMARK_NAMES[destination]}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=_INK)
    return ax


def _history_summary(runs, metric, window):
    histories = [frame.set_index("episode")[metric].astype(float).rolling(
        window, min_periods=1).mean() for frame in runs]
    if not histories:
        raise ValueError("Each curve requires at least one seed history")
    aligned = pd.concat(histories, axis=1)
    mean = aligned.mean(axis=1)
    se = aligned.std(axis=1, ddof=1).fillna(0) / np.sqrt(aligned.count(axis=1))
    return aligned.index.to_numpy(), mean.to_numpy(), se.to_numpy()


def _curve(ax, runs, metric, window, label, color):
    episodes, mean, se = _history_summary(runs, metric, window)
    lower, upper = mean-se, mean+se
    if metric == "success":
        lower, upper = np.maximum(lower,0), np.minimum(upper,1)
    elif metric in ("steps", "illegal_actions"):
        lower = np.maximum(lower,0)
    ax.plot(episodes,mean,color=color,label=str(label),linewidth=1.8)
    ax.fill_between(episodes,lower,upper,color=color,alpha=.16,linewidth=0)


def _return_scale(ax):
    ax.set_yscale("symlog", linthresh=100)
    lower, upper = sorted(ax.get_ylim())
    candidates = (-100000,-10000,-3000,-1000,-500,-200,-100,
                  -80,-60,-40,-20,0,10,20,40,60,80,100)
    ticks = [v for v in candidates if lower <= v <= upper]
    ax.set_yticks(ticks,[f"{v:g}" for v in ticks])


def _metric_axis(ax, metric):
    if metric == "return":
        _return_scale(ax)
        ax.set_ylabel("Episode return (symlog; linear within ±100)",fontsize=8.5)
    elif metric == "steps":
        ax.set_yscale("log")
        ax.set_ylabel("Steps per episode (log)",fontsize=9)
    elif metric == "success":
        ax.set_ylim(-.025,1.045)
        ax.set_yticks(np.linspace(0,1,6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Episodes reaching the goal",fontsize=9)
    else:
        if ax.get_ylim()[1] > 1:
            ax.set_yscale("symlog",linthresh=1)
            ax.set_ylabel("Illegal actions (symlog; linear below 1)",fontsize=8.5)
        else:
            ax.set_ylabel("Illegal actions per episode",fontsize=9)
        ax.set_ylim(bottom=0)


def plot_learning(groups, window=100):
    """Means and ±1 SE of independently trained seeds; all episodes retained."""
    fig, axes = plt.subplots(2,2,figsize=(13.5,8.4),facecolor=_PAPER)
    specs = (("return","Episode return"),("success","Successful deliveries"),
             ("steps","Trip length"),("illegal_actions","Illegal pickup / dropoff"))
    for ax,(metric,title) in zip(axes.flat,specs):
        _axis_style(ax)
        for index,(label,runs) in enumerate(groups.items()):
            _curve(ax,runs,metric,window,label,_PALETTE[index%len(_PALETTE)])
        _metric_axis(ax,metric)
        ax.set_title(title,loc="left",fontsize=12,color=_INK,weight="bold",pad=10)
        ax.set_xlabel("Training episode",fontsize=9)
        ax.margins(x=.01)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle("A taxi that learns from experience",x=.065,y=.98,
                 ha="left",fontsize=20,color=_INK,weight="bold")
    fig.text(.065,.925,f"Trailing {window}-episode averages  •  Shading: ±1 SE across independent training seeds",
             fontsize=9.5,color=_MUTED)
    fig.legend(handles,labels,loc="upper right",bbox_to_anchor=(.98,.985),
               ncol=min(len(labels),3),frameon=False,fontsize=9)
    counts="; ".join(f"{label}: {len(runs)} seeds" for label,runs in groups.items())
    fig.text(.065,.025,counts+". Early failures are retained; nonlinear scales keep late learning visible.",
             fontsize=8.5,color=_MUTED)
    fig.subplots_adjust(left=.085,right=.98,top=.86,bottom=.105,hspace=.39,wspace=.27)
    return fig


def plot_sweep(groups, param, window=100):
    """One parameter varied at a time; legend occupies a dedicated row."""
    fig,axes=plt.subplots(1,3,figsize=(14,5.1),facecolor=_PAPER)
    symbols={"alpha":"α","gamma":"γ","epsilon":"ε"}
    for ax,(metric,title) in zip(axes,(("return","Episode return"),
                                      ("success","Successful deliveries"),
                                      ("illegal_actions","Illegal actions"))):
        _axis_style(ax)
        for index,(label,runs) in enumerate(groups.items()):
            text=str(label)
            if param in symbols and "=" not in text:
                text=f"{symbols[param]} = {text}"
            _curve(ax,runs,metric,window,text,_PALETTE[index%len(_PALETTE)])
        _metric_axis(ax,metric)
        ax.set_title(title,loc="left",fontsize=11.5,color=_INK,weight="bold",pad=10)
        ax.set_xlabel("Training episode",fontsize=9)
        ax.margins(x=.01)
    display=symbols.get(param,param)
    title=f"How {param} ({display}) changes learning" if param in symbols else f"Learning under different {param} settings"
    fig.suptitle(title,x=.065,y=.98,ha="left",fontsize=18,color=_INK,weight="bold")
    fig.text(.065,.88,f"One factor varied  •  {window}-episode averages  •  Shading: ±1 SE across seeds",
             fontsize=9,color=_MUTED)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper right",bbox_to_anchor=(.99,.82),
               ncol=min(4,len(labels)),frameon=False,fontsize=9)
    fig.text(.065,.025,"Returns: symlog threshold 100. Illegal actions: symlog threshold 1 when needed. Success uses a linear percentage scale.",
             fontsize=8,color=_MUTED)
    fig.subplots_adjust(left=.075,right=.98,top=.67,bottom=.17,wspace=.31)
    return fig


def _short_mode(mode):
    text=str(mode).lower()
    if "greedy" in text:
        return "Greedy"
    if "explor" in text or "epsilon" in text:
        return "Explore"
    if "random" in text:
        return "Random"
    if "oracle" in text or "optimal" in text:
        return "Oracle"
    return str(mode).split("(",1)[0].strip()


def plot_evaluation(df):
    """Average tasks within each seed, then summarize independent seed replicates.

    Random and planning references may appear as agents alongside trained agents.
    Missing agent/mode combinations remain empty rather than becoming zero bars.
    """
    needed={"agent","mode","train_seed","return","success","illegal_actions"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing evaluation columns: {sorted(needed-set(df.columns))}")
    metrics=["return","success","illegal_actions"]
    per_seed=df.groupby(["agent","mode","train_seed"],sort=False,observed=True)[metrics].mean().reset_index()
    agents=list(dict.fromkeys(per_seed.agent))
    modes=list(dict.fromkeys(per_seed["mode"]))
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),facecolor=_PAPER)
    width=.78/max(len(agents),1)
    x=np.arange(len(modes))
    for ax,(metric,title) in zip(axes,(("return","Return during frozen evaluation"),
                                      ("success","Successful deliveries"),
                                      ("illegal_actions","Illegal actions"))):
        _axis_style(ax)
        for index,agent in enumerate(agents):
            means,errors=[],[]
            for mode in modes:
                values=per_seed.loc[(per_seed.agent==agent)&(per_seed["mode"]==mode),metric]
                means.append(values.mean() if len(values) else np.nan)
                errors.append(values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.)
            positions=x+(index-(len(agents)-1)/2)*width
            bars=ax.bar(positions,means,width*.88,color=_PALETTE[index%len(_PALETTE)],
                        label=agent,yerr=errors,capsize=3,edgecolor="white",linewidth=.5,
                        error_kw={"elinewidth":1,"ecolor":_INK},zorder=3)
            for bar,mean,error in zip(bars,means,errors):
                if not np.isfinite(mean):
                    continue
                label=f"{mean:.0%}" if metric=="success" else (
                    f"{mean:.3f}" if metric=="illegal_actions" and 0<mean<1 else f"{mean:.1f}")
                offset=5+10*(index%2) if metric=="success" else (5 if mean>=0 else -5)
                outer=mean+error if mean>=0 else mean-error
                ax.annotate(label,(bar.get_x()+bar.get_width()/2,outer),
                            xytext=(0,offset),textcoords="offset points",ha="center",
                            va="bottom" if mean>=0 else "top",fontsize=7,color=_INK)
        ax.set_xticks(x,[_short_mode(mode) for mode in modes],fontsize=8.5)
        ax.set_title(title,loc="left",fontsize=11.5,color=_INK,weight="bold",pad=12)
        ax.axhline(0,color="#B6C9D2",linewidth=.8)
        ax.margins(y=.23,x=.06)
        if metric=="return":
            _return_scale(ax)
            ax.set_ylabel("Mean return (symlog; linear within ±100)",fontsize=8.5)
        elif metric=="success":
            ax.set_ylim(0,1.22)
            ax.set_yticks(np.linspace(0,1,6))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Success rate",fontsize=9)
        else:
            if ax.get_ylim()[1]>1:
                ax.set_yscale("symlog",linthresh=1)
                ax.set_ylabel("Mean illegal actions (symlog; linear below 1)",fontsize=8.2)
            else:
                ax.set_ylabel("Mean illegal actions per trip",fontsize=9)
            ax.set_ylim(bottom=0)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.suptitle("How do the frozen Taxi policies perform?",x=.065,y=.98,
                 ha="left",fontsize=18,color=_INK,weight="bold")
    fig.text(.065,.89,"Frozen policies  •  Bars: mean ±1 SE across independent seed replicates",
             fontsize=9,color=_MUTED)
    fig.legend(handles,labels,loc="upper right",bbox_to_anchor=(.99,.83),
               ncol=min(len(labels),4),frameon=False,fontsize=8.5)
    fig.text(.065,.025,"Trips are averaged within a training seed (or reference replica) first. Missing agent/mode combinations are left empty.",
             fontsize=8.3,color=_MUTED)
    fig.subplots_adjust(left=.075,right=.98,top=.68,bottom=.17,wspace=.31)
    return fig


def _slice_values(env,q,passenger,destination):
    values=np.full((env.rows,env.cols),np.nan)
    actions=np.full((env.rows,env.cols),-1,dtype=int)
    if passenger==destination:
        return values,actions
    for row in range(env.rows):
        for col in range(env.cols):
            state=int(env.encode(row,col,int(passenger),int(destination)))
            values[row,col]=np.max(q[state])
            actions[row,col]=int(np.argmax(q[state]))
    return values,actions


def plot_policy_slices(env,q,passenger=0,destination=3):
    """Two conditioned slices of the full 500-state Q-table, with real road walls."""
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.7),facecolor=_PAPER)
    contexts=[(passenger,f"Passenger waiting at {_LANDMARK_NAMES[passenger]}"),
              (4,"Passenger on board")]
    for ax,(p,title) in zip(axes,contexts):
        values,actions=_slice_values(env,q,p,destination)
        _board(ax,env,title=f"{title}  |  destination {_LANDMARK_NAMES[destination]}",
               passenger=p,destination=destination,values=values,actions=actions)
        if p==destination:
            ax.text(2,2,"No waiting-start decisions:\npassenger already at destination",ha="center",va="center",
                    fontsize=11,color=_INK,bbox=dict(facecolor="white",edgecolor="none",alpha=.9),zorder=25)
    fig.suptitle("What the learned Q-table says to do",x=.065,y=.98,
                 ha="left",fontsize=18,color=_INK,weight="bold")
    fig.text(.065,.045,"Number: maxₐ Q(s,a). Arrow: one maximizing action (first tie shown); actual rollouts break ties randomly. P = pickup; D = dropoff.",
             fontsize=8.4,color=_MUTED)
    fig.text(.065,.018,"Each board holds passenger status and destination fixed while varying taxi position. Terminal/unreachable waiting contexts are masked; color ranges are per board.",
             fontsize=8,color=_MUTED)
    fig.subplots_adjust(left=.065,right=.975,top=.86,bottom=.16,wspace=.20)
    return fig


def plot_checkpoint(df):
    """Seed-wise checkpoint means and SE for success, optimality and Q error."""
    required={"condition","seed","episode","success","optimal_rate","mean_return","q_mae"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing checkpoint columns: {sorted(required-set(df.columns))}")
    fig,axes=plt.subplots(1,3,figsize=(14,5.1),facecolor=_PAPER)
    conditions=list(dict.fromkeys(df.condition))
    for ax,(metric,title) in zip(axes,(("success","Successful deliveries"),
                                      ("optimal_rate","Trips using the fewest steps"),
                                      ("q_mae","Q-value error against planning"))):
        _axis_style(ax)
        for index,condition in enumerate(conditions):
            selected=df[df.condition==condition]
            grouped=selected.groupby("episode")[metric]
            mean=grouped.mean(); se=grouped.std(ddof=1).fillna(0)/np.sqrt(grouped.count())
            x=mean.index.to_numpy(); y=mean.to_numpy(); error=se.to_numpy()
            ax.plot(x,y,marker="o",markersize=4,linewidth=1.8,label=condition,
                    color=_PALETTE[index%len(_PALETTE)])
            lower=y-error; upper=y+error
            if metric in ("success","optimal_rate"):
                lower=np.maximum(lower,0); upper=np.minimum(upper,1)
            ax.fill_between(x,lower,upper,color=_PALETTE[index%len(_PALETTE)],alpha=.16)
        ax.set_xscale("symlog",linthresh=10)
        ax.set_title(title,loc="left",fontsize=11,color=_INK,weight="bold",pad=11)
        ax.set_xlabel("Training episode (symlog; linear through 10)",fontsize=8.3)
        if metric in ("success","optimal_rate"):
            ax.set_ylim(-.025,1.045)
            ax.set_yticks(np.linspace(0,1,6))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Mean absolute Q error",fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.suptitle("Checking learning against an exact reference",x=.065,y=.98,
                 ha="left",fontsize=18,color=_INK,weight="bold")
    fig.text(.065,.88,"Frozen checkpoints on independent evaluation trips  •  Shading: ±1 SE across independent training seeds",
             fontsize=9,color=_MUTED)
    fig.legend(handles,labels,loc="upper right",bbox_to_anchor=(.99,.82),
               ncol=min(len(labels),4),frameon=False,fontsize=8.5)
    fig.subplots_adjust(left=.075,right=.98,top=.67,bottom=.16,wspace=.29)
    return fig


def animate_taxi(env,trajectory,interval=300):
    """Animate recorded states/actions/rewards; no transitions or rewards invented."""
    states=list(trajectory["states"])
    rewards=list(trajectory["rewards"])
    actions=list(trajectory["actions"])
    if not states or len(states)!=len(rewards)+1 or len(actions)!=len(rewards):
        raise ValueError("Trajectory needs initial state + one state, action and reward per transition")
    fig,ax=plt.subplots(figsize=(7.1,7.5),facecolor=_PAPER)
    fig.subplots_adjust(left=.11,right=.93,top=.80,bottom=.17)
    fig.suptitle("A learned taxi trip, one decision at a time",x=.07,y=.97,
                 ha="left",fontsize=16,color=_INK,weight="bold")
    status=fig.text(.07,.88,"",fontsize=10,color=_INK)
    fig.text(.07,.035,"Passenger: purple square  •  Destination: flag  •  Orange: before pickup  •  Teal: on board",
             fontsize=8,color=_MUTED)

    def update(frame):
        step=min(frame,len(states)-1)
        ax.clear()
        current=states[step]
        _,_,p,d=map(int,env.decode(int(current)))
        action="Start" if step==0 else _ACTION_NAMES[int(actions[step-1])]
        cumulative=float(np.sum(rewards[:step]))
        illegal=int(np.count_nonzero(np.isclose(rewards[:step],-10)))
        finished=step==len(states)-1
        outcome="  |  DELIVERED" if finished and p==d else ("  |  rollout ended" if finished else "")
        status.set_text(f"Step {step}  |  action: {action}  |  return {cumulative:.0f}  |  illegal {illegal}{outcome}")
        draw_taxi_map(ax,env,current,title="Recorded rollout",
                      path={"states":states[:step+1],"actions":actions[:step],"rewards":rewards[:step]})
        return [status]

    anim=animation.FuncAnimation(fig,update,frames=len(states)+5,
                                  interval=interval,blit=False,repeat=True)
    plt.close(fig)
    return anim
