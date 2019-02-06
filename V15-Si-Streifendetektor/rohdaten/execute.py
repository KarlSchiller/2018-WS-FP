import Sourcescan_ana as RSscan
import Pedestal_ana as Ped
import CCEL_ana as CCEL
import CCEQ_ana as CCEQ
import Laserscan_ana as Laser

print "Converting Pedestal file"
Ped.main()
print "Extracting Laserscan data"
Laser.main()
print "Extracting CCE-Laser data"
CCEL.main()
print "Extracting CCE-Quelle data"
CCEQ.main()
print "Clustering and Analyzing of RS-Run"
RSscan.main()
