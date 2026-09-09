"""Draw the field maps from a newly built sample and its source parcel context."""
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.plotting import plot_polygon

LABELS = {'above_prediction': 'Above prediction', 'below_prediction': 'Below prediction',
          'mixed': 'Mixed', 'close_to_prediction': 'Close to prediction'}
COLORS = ['#b2182b', '#2166ac', '#8c4fb1', '#3f8e4f']


def render(out, sites, members, cores, walks, starts, parks, background,
           parcel_geometry, roads):
    tree = shapely.STRtree(background)
    road_geometry = np.array(list(roads.values()), dtype=object)
    road_tree = shapely.STRtree(road_geometry)
    norm, cmap = Normalize(-.6, .6), plt.get_cmap('RdBu_r')

    def polygon(ax, geometry, **style):
        for part in shapely.get_parts(geometry):
            if part.geom_type == 'Polygon':
                plot_polygon(part, ax=ax, add_points=False, **style)

    def line(ax, geometry, **style):
        for part in shapely.get_parts(geometry):
            xy = shapely.get_coordinates(part)
            if len(xy):
                ax.plot(xy[:, 0], xy[:, 1], **style)

    def draw(ax, i, context=True, numbers=False):
        site = sites.iloc[i]
        group = members[members.circuit_id.eq(site.circuit_id)]
        assessed = group[group.assessment_required]
        shape = shapely.union_all([parcel_geometry[k] for k in assessed.parcel_id])
        if context:
            shape = shapely.union_all([shape, cores[i], walks[i]])
        x0, y0, x1, y1 = shape.bounds
        width = max(x1-x0, y1-y0)+100
        cx, cy = (x0+x1)/2, (y0+y1)/2
        box = shapely.box(cx-width/2, cy-width/2, cx+width/2, cy+width/2)
        for g in background[tree.query(box, predicate='intersects')]:
            polygon(ax, g, facecolor='#f5f5f5', edgecolor='#d4d4d4', linewidth=.3)
        polygon(ax, shapely.intersection(parks[i], box),
                facecolor='#bfe0ad', edgecolor='#438242', linewidth=1.1)
        for g in road_geometry[road_tree.query(box, predicate='intersects')]:
            line(ax, g, color='#bbbbbb', linewidth=.7)
        for row in group.itertuples():
            g = parcel_geometry[row.parcel_id]
            color = cmap(norm(row.relative_departure)) if row.assessment_required and np.isfinite(row.relative_departure) else '#999999'
            polygon(ax, g, facecolor=color, edgecolor='#333333', linewidth=.7)
            if numbers and row.assessment_required:
                point = g.representative_point()
                ax.text(point.x, point.y, str(row.visit_order), fontsize=5.5,
                        ha='center', va='center')
        line(ax, cores[i], color='black', linewidth=2)
        line(ax, walks[i], color='#635198', linewidth=1.4, linestyle='--')
        for part in shapely.get_parts(shapely.union_all([cores[i], walks[i]])):
            if part.length < 35:
                continue
            for fraction, direction in [(.35, 1), (.65, -1)]:
                a, b = [part.interpolate(part.length*fraction+d) for d in [-6, 6]]
                vector = np.array([b.x-a.x, b.y-a.y])
                vector /= max(np.linalg.norm(vector), .001)
                offset = direction*4*np.array([-vector[1], vector[0]])
                first, last = np.array([a.x, a.y])+offset, np.array([b.x, b.y])+offset
                if direction < 0:
                    first, last = last, first
                ax.annotate('', xy=last, xytext=first,
                            arrowprops=dict(arrowstyle='->', color='#513777', lw=.9))
        ax.scatter([starts[i].x], [starts[i].y], s=30, color='#e5a000',
                   edgecolor='black', zorder=10)
        ax.set(xlim=(cx-width/2, cx+width/2), ylim=(cy-width/2, cy+width/2), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(.98, .98, 'N ↑', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        scale = 50 if width < 400 else 100
        ax.plot([cx-width*.43, cx-width*.43+scale], [cy-width*.43]*2, color='black', linewidth=2)
        ax.text(cx-width*.43, cy-width*.40, f'{scale} m', fontsize=7)
        ax.set_title(f'{site.street_name} | {site.assessment_parcels} parcels', fontsize=10)

    fig, axes = plt.subplots(3, 4, figsize=(15, 11), layout='constrained')
    for col, category in enumerate(LABELS):
        primary = sites[sites.category.eq(category)].sort_values('rank_in_category').head(3)
        for row, i in enumerate(primary.index):
            draw(axes[row, col], i)
            axes[row, col].set_title(
                f'{LABELS[category]} #{row+1}\n{sites.loc[i, "street_name"]}', fontsize=10)
    fig.suptitle('Primary field circuits', fontsize=15)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 shrink=.7, label='Relative departure from predicted water use', extend='both')
    fig.savefig(out/'top_ranked_streets.png', dpi=160)
    plt.close(fig)
    metadata = {'Title': 'Reviewed field circuits', 'CreationDate': None, 'ModDate': None}
    with PdfPages(out/'ranked_field_review_maps.pdf', metadata=metadata) as pdf:
        for category in LABELS:
            for i in sites[sites.category.eq(category)].sort_values('rank_in_category').index:
                s = sites.loc[i]
                fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
                draw(axes[0], i, context=False, numbers=True)
                draw(axes[1], i)
                axes[0].set_xlabel('Numbered assessment sequence')
                axes[1].set_xlabel('Purple: park connection and return directions; green: paired space')
                fig.suptitle(f'{s.circuit_id} — {LABELS[category]} #{s.rank_in_category} — {s.priority.upper()}')
                text = (f'{s.assessment_parcels} assessment parcels | Approximate return walk: {s.estimated_walk_m:.0f} m\n'
                        f'Paired space: {s.park_name} (parcel {s.park_id}) | {s.park_area_m2:,.0f} m²\n'
                        f'Suggested start: {s.latitude:.6f}, {s.longitude:.6f}\n'
                        f'Above +0.20: {s.above_fraction:.0%} | Below -0.20: {s.below_fraction:.0%} | Close: {s.close_fraction:.0%}\n'
                        'Red: above prediction; blue: below; gray: missing model or context. Model year: 2021.\n'
                        'Yellow: suggested start. Routes are schematic; confirm parking and entrances in the field.')
                fig.text(.06, .06, text, fontsize=9)
                fig.subplots_adjust(bottom=.27, top=.88, wspace=.1)
                pdf.savefig(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    for category, color in zip(LABELS, COLORS):
        indices = sites.index[sites.category.eq(category)]
        xy = shapely.get_coordinates([cores[i].centroid for i in indices])
        ax.scatter(xy[:, 0], xy[:, 1], s=28, c=color, label=LABELS[category])
        for i, (x, y) in zip(indices, xy):
            ax.text(x+25, y+25, str(sites.loc[i, 'rank_in_category']), fontsize=7)
    ax.set(aspect='equal', xlabel='UTM easting (m)', ylabel='UTM northing (m)',
           title=f'{len(sites)} field circuits — numbers are ranks within category')
    ax.legend()
    fig.savefig(out/'all_sites_overview.png', dpi=160)
    plt.close(fig)
