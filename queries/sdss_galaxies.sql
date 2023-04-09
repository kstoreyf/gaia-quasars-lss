SELECT
specObjID, ra, dec into mydb.MyTable from SpecObj
WHERE class='GALAXY' AND subClass!='AGN' AND subClass!='AGN BROADLINE' AND zWarning=0
