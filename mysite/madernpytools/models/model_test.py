import madernpytools.models.toolset_model as mts
import madernpytools.madern_widgets as mwids


if __name__=="__main__":
    bb_test = mts.ModelItemLoader().load('../data/library/RiRo1250_bb.xml')

    mwids.generate_editors_in_tab(names=['BearingBlock'])
    mwids.ModelEditor(mts.BearingBlock)

