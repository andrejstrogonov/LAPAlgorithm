class LPStructure:
    def __init__(self, eLengthItems, eItemBitSize):
        self.eLengthItems = eLengthItems
        self.eItemBitSize = eItemBitSize

    def EQ(self, ls, rs):
        for i in range(self.eLengthItems):
            if ls[i] != rs[i]:
                return False
        return True

    def EZ(self, ls):
        for i in range(self.eLengthItems):
            if ls[i]:
                return False
        return True

    def LE(self, ls, rs):
        for i in range(self.eLengthItems):
            if (ls[i] | rs[i]) != rs[i]:
                return False
        return True

    def LT(self, ls, rs):
        bExistLT = False
        for i in range(self.eLengthItems):
            if (ls[i] | rs[i]) == rs[i]:
                if ls[i] != rs[i]:
                    bExistLT = True
            else:
                return False
        return bExistLT

    def lJoin(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] |= rs[i]

    def lMeet(self, ls, rs):
        for i in range(self.eLengthItems):
            ls[i] &= rs[i]

    def lDiff(self, ls, rs):
        res = False
        for i in range(self.eLengthItems):
            if ls[i] & rs[i]:
                ls[i] &= ~rs[i]
                res = True
        return res

    def isMeet(self, ls, rs):
        for i in range(self.eLengthItems):
            if ls[i] & rs[i]:
                return True
        return False

    def isON(self, eTest, nAtom):
        nItem = nAtom // self.eItemBitSize
        nBit = nAtom % self.eItemBitSize
        nMask = 1 << (self.eItemBitSize - 1 - nBit)
        return bool(eTest[nItem] & nMask)

import unittest

class TestLPStructure(unittest.TestCase):

    def setUp(self):
        self.lp = LPStructure(eLengthItems=4, eItemBitSize=8)

    def test_EQ(self):
        self.assertTrue(self.lp.EQ([0, 0, 0, 0], [0, 0, 0, 0]))
        self.assertFalse(self.lp.EQ([1, 0, 0, 0], [0, 0, 0, 0]))

    def test_EZ(self):
        self.assertTrue(self.lp.EZ([0, 0, 0, 0]))
        self.assertFalse(self.lp.EZ([1, 0, 0, 0]))

    def test_LE(self):
        self.assertTrue(self.lp.LE([0, 0, 0, 0], [1, 1, 1, 1]))
        self.assertFalse(self.lp.LE([1, 0, 0, 0], [0, 0, 0, 0]))

    def test_LT(self):
        self.assertTrue(self.lp.LT([0, 0, 0, 0], [1, 1, 1, 1]))
        self.assertFalse(self.lp.LT([1, 1, 1, 1], [1, 1, 1, 1]))

    def test_lJoin(self):
        ls = [0, 0, 0, 0]
        rs = [1, 1, 1, 1]
        self.lp.lJoin(ls, rs)
        self.assertEqual(ls, [1, 1, 1, 1])

    def test_lMeet(self):
        ls = [1, 1, 1, 1]
        rs = [1, 0, 1, 0]
        self.lp.lMeet(ls, rs)
        self.assertEqual(ls, [1, 0, 1, 0])

    def test_lDiff(self):
        ls = [1, 1, 1, 1]
        rs = [1, 0, 1, 0]
        result = self.lp.lDiff(ls, rs)
        self.assertTrue(result)
        self.assertEqual(ls, [0, 1, 0, 1])

    def test_isMeet(self):
        self.assertTrue(self.lp.isMeet([1, 1, 1, 1], [1, 0, 1, 0]))
        self.assertFalse(self.lp.isMeet([0, 0, 0, 0], [1, 1, 1, 1]))

    def test_isON(self):
        eTest = [0b10000000, 0b00000000, 0b00000000, 0b00000000]
        self.assertTrue(self.lp.isON(eTest, 0))
        self.assertFalse(self.lp.isON(eTest, 8))

if __name__ == "__main__":
   unittest.main()