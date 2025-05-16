Peptide insertion
=================

The extent of peptide insertion into the membrane was assessed using the Z-position coordinates of the peptides relative to the plane of the phosphorus atoms
in the lipid head-groups of the upper membrane leaflet. The Z-position of a peptide was defined by the z coordinate of the residue with the greatest insertion, corresponding to the minimum z value. When visual inspection of the simulation trajectory indicated negligible membrane curvature, the membrane plane was defined
as the mean z value of the phosphorus atoms in the lipid headgroups of the upper
leaflet.

However, more pronounced membrane curvature, as seen in Chapter 5 made this approach less accurate. Instead, the membrane plane level relative to each residue was computed using the $z$-position of the closest phosphorous atom in the lipid headgroups The closest phosphorous atoms of lipid headgroups to each residue were computed using the KDTree algorithm from the Scipy library. The KDTree algorithm was first introduced in 1975 and uses a binary tree to split spatial data. Each node represents an axis and splits the data points based on whether their coordinate along that axis is greater than or less than a specific value. This algorithm identifies the closest phosphorus atom to each residue and its coordinates, which are then used to compute the Euclidean distance between the residue and the corresponding phosphorus atom.